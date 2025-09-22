// admin.js - Admin Task Management (refactored for readability)

(function () {
  document.addEventListener('DOMContentLoaded', initAdminTaskPage);

  function initAdminTaskPage() {
    // Fallback getApiUrl if not globally defined
    if (typeof window.getApiUrl !== 'function') {
      window.getApiUrl = function (path) { return path; };
    }

    const tableBody = document.querySelector('#admin-task-table tbody');
    const loading = document.getElementById('admin-loading');
    const refreshBtn = document.getElementById('refresh-btn');
    const clearSelectedBtn = document.getElementById('clear-selected-btn');
    const searchInput = document.getElementById('admin-search');
    const searchType = document.getElementById('admin-search-type');
    const clearSearchBtn = document.getElementById('clear-admin-search-btn');

    if (!tableBody) {
      console.error('Admin task table body not found.');
      return;
    }

    let tasks = [];
    let currentSort = { key: 'created_at', dir: 'desc' };

    // ---------------- UI Helpers ----------------
    function showLoading(show) {
      if (loading) {
        loading.style.display = show ? 'flex' : 'none';
      }
    }

    function escapeHtml(str) {
      return (str || '').replace(/[&<>"']/g, function (c) {
        return ({
          '&': '&amp;',
          '<': '&lt;',
          '>': '&gt;',
          '"': '&quot;',
          "'": '&#39;'
        })[c];
      });
    }

    function shorten(str) {
      if (!str) return '';
      return str.length > 40 ? str.slice(0, 37) + '...' : str;
    }

    function parseDate(str) {
      const m = str.match(/^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})/);
      if (m) {
        return new Date(m[1], m[2] - 1, m[3], m[4], m[5], m[6]).getTime();
      }
      return null;
    }

    function compare(a, b, key, asc) {
      const vaRaw = a[key] ?? '';
      const vbRaw = b[key] ?? '';
      const va = vaRaw.toString().toLowerCase();
      const vb = vbRaw.toString().toLowerCase();

      const da = parseDate(va);
      const db = parseDate(vb);
      if (da && db) {
        return asc ? da - db : db - da;
      }
      return asc
        ? va.localeCompare(vb, undefined, { numeric: true, sensitivity: 'base' })
        : vb.localeCompare(va, undefined, { numeric: true, sensitivity: 'base' });
    }

    function renderStatus(status) {
      const cls = 'status-badge status-' + (status || '').replace(/\s+/g, '');
      return '<span class="' + cls + '">' + escapeHtml(status || '') + '</span>';
    }

    function renderStatusBadge(status) {
      if (!status) {
        return '<span class="badge bg-secondary">-</span>';
      }
      
      switch (status) {
        case 'Completed':
          return '<span class="badge bg-success">Completed</span>';
        case 'Exported':
          return '<span class="badge bg-warning">Exported</span>';
        case 'Pending':
          return '<span class="badge bg-progress">Pending</span>';
        case 'Hold':
          return '<span class="badge bg-progress-light">Hold</span>';
        case 'Deleted':
          return '<span class="badge bg-secondary">Deleted</span>';
        default:
          return '<span class="badge bg-secondary">' + escapeHtml(status) + '</span>';
      }
    }

    function formatDateTime(dateStr) {
      if (!dateStr && dateStr !== 0) return '';
      try {
        let date;
        // Accept Date objects
        if (dateStr instanceof Date) {
          date = dateStr;
        } else {
          const s = String(dateStr).trim();
          // If it's a plain number (epoch ms or s), construct Date
          if (/^\d+$/.test(s)) {
            // If length looks like seconds (10 digits), convert to ms
            date = new Date(s.length === 10 ? Number(s) * 1000 : Number(s));
          } else {
            // Convert common 'YYYY-MM-DD HH:MM:SS' to ISO-like 'YYYY-MM-DDTHH:MM:SS' for reliable parsing
            const t = s.replace(/^([0-9]{4}-[0-9]{2}-[0-9]{2})\s+([0-9]{2}:[0-9]{2}:[0-9]{2})/, '$1T$2');
            date = new Date(t);
          }
        }
        if (isNaN(date.getTime())) return String(dateStr);
        const pad = function (n) { return String(n).padStart(2, '0'); };
        return date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' + pad(date.getDate()) + ' ' + pad(date.getHours()) + ':' + pad(date.getMinutes()) + ':' + pad(date.getSeconds());
      } catch (e) {
        return String(dateStr);
      }
    }

    function buildActions(task) {
      const canDelete = task.status_ccx !== 'Deleted';
      const canRepend = task.status_ccx === 'Exported' || task.status_ccx === 'Deleted';
      return [
        '<button class="action-btn btn-delete delete-btn" data-task-id="' + task.task_id + '" ' + (canDelete ? '' : 'disabled') + '>Drop</button>',
        '<button class="action-btn btn-repend repend-btn" data-task-id="' + task.task_id + '" ' + (canRepend ? '' : 'disabled') + '>Repend</button>'
      ].join('\n');
    }

    function showNoTasksRow(message) {
      tableBody.innerHTML = '';
      const tr = document.createElement('tr');
      const td = document.createElement('td');
  td.colSpan = 11;
      td.style.textAlign = 'center';
      td.textContent = message;
      tr.appendChild(td);
      tableBody.appendChild(tr);
    }

    // ---------------- Data Fetch ----------------
    async function fetchTasks() {
      showLoading(true);
      try {
        const url = getApiUrl('/admin/tasks');
        const res = await fetch(url, {
          headers: { 'Accept': 'application/json' }
        });

        if (!res.ok) {
          throw new Error('HTTP ' + res.status + ' ' + res.statusText);
        }

        const data = await res.json();
        if (!data.success) {
          throw new Error(data.message || 'Failed to load tasks');
        }

        tasks = Array.isArray(data.tasks) ? data.tasks : [];
        if (!tasks.length) {
          showNoTasksRow('No tasks found.');
        } else {
          render();
        }
      } catch (e) {
        console.error('Fetch tasks error:', e);
        showNoTasksRow('Error loading tasks: ' + e.message);
      } finally {
        showLoading(false);
      }
    }

    // ---------------- Rendering ----------------
    function render() {
      const term = (searchInput.value || '').toLowerCase();
      const type = searchType.value;

      let filtered = tasks.filter(function (t) {
        if (!term) return true;
        if (type === 'all') {
          return Object.values(t).some(function (v) {
            if (v === null || v === undefined) return false;
            return String(v).toLowerCase().includes(term);
          });
        }
        const keyMap = {
          task_id: 'task_id',
          filename: 'filename',
          with_error: 'with_error',
          status_ccx: 'status_ccx',
          status_infor: 'status_infor',
          wrike_id: 'wrike_task_id',
          user: 'user_name'
        };
        const k = keyMap[type];
        const val = t[k];
        return val !== null && val !== undefined && String(val).toLowerCase().includes(term);
      });

      filtered.sort(function (a, b) {
        return compare(a, b, currentSort.key, currentSort.dir === 'asc');
      });

      tableBody.innerHTML = '';

      if (!filtered.length) {
        showNoTasksRow('No tasks match your filter.');
        return;
      }

      filtered.forEach(function (task) {
        const tr = document.createElement('tr');
        if (task.status_ccx === 'Deleted') {
          tr.classList.add('row-deleted');
        }

        const clearCellContent = task.status_ccx === 'Deleted'
          ? '<input type="checkbox" class="clear-checkbox" data-task-id="' + task.task_id + '" />'
          : '';

        // Render with_error as a badge-like indicator (Yes/No or show raw value)
        function renderWithError(val) {
          if (val === null || val === undefined || val === '') return '<span class="badge bg-secondary">-</span>';
          const str = String(val).toLowerCase();
          if (str === 'true' || str === '1' || str === 'yes') return '<span class="badge bg-danger">Yes</span>';
          if (str === 'false' || str === '0' || str === 'no') return '<span class="badge bg-success">No</span>';
          return '<span class="badge bg-secondary">' + escapeHtml(String(val)) + '</span>';
        }

        tr.innerHTML = [
          '<td>' + task.task_id + '</td>',
          '<td>' + escapeHtml(task.user_name || '') + '</td>',
          '<td>' + (task.wrike_task_id || '-') + '</td>',
          '<td class="filename-cell" title="' + escapeHtml(task.filename || '') + '">' + escapeHtml(shorten(task.filename)) + '</td>',
          '<td>' + renderWithError(task.with_error) + '</td>',
          '<td>' + renderStatusBadge(task.status_ccx) + '</td>',
          '<td>' + renderStatusBadge(task.status_infor) + '</td>',
          '<td>' + formatDateTime(task.created_at) + '</td>',
          '<td>' + formatDateTime(task.updated_at) + '</td>',
          '<td>' + buildActions(task) + '</td>',
          '<td>' + clearCellContent + '</td>'
        ].join('\n');

        tableBody.appendChild(tr);
      });

      attachRowHandlers();
      updateClearSelectedState();
    }

    // ---------------- Actions ----------------
    function updateClearSelectedState() {
      const anyChecked = tableBody.querySelectorAll('.clear-checkbox:checked').length > 0;
      clearSelectedBtn.disabled = !anyChecked;
    }

    async function clearSelected() {
      const ids = Array.from(tableBody.querySelectorAll('.clear-checkbox:checked'))
        .map(function (cb) { return cb.dataset.taskId; });

      if (!ids.length) return;

      if (!confirm('Permanently clear ' + ids.length + ' deleted task(s)? This cannot be undone.')) {
        return;
      }

      await postAction('/admin/tasks/clear_deleted', { task_ids: ids });
      await fetchTasks();
    }

    async function postAction(url, payload) {
      showLoading(true);
      try {
        const res = await fetch(getApiUrl(url), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(payload || {})
        });

        const data = await res.json().catch(function () { return {}; });
        if (!res.ok || !data.success) {
            throw new Error(data.message || ('Request failed: ' + res.status));
        }
        return data;
      } catch (e) {
        alert('Error: ' + e.message);
        console.error('Action error for', url, e);
      } finally {
        showLoading(false);
      }
    }

    function attachRowHandlers() {
      tableBody.querySelectorAll('.delete-btn').forEach(function (btn) {
        btn.addEventListener('click', async function () {
          if (!confirm('Permanently delete this task? This cannot be undone.')) return;
          await postAction('/admin/task/' + btn.dataset.taskId + '/delete');
          await fetchTasks();
        });
      });

      tableBody.querySelectorAll('.repend-btn').forEach(function (btn) {
        btn.addEventListener('click', async function () {
          if (!confirm('Revert this exported task back to Pending?')) return;
          await postAction('/admin/task/' + btn.dataset.taskId + '/repend');
          await fetchTasks();
        });
      });

      tableBody.querySelectorAll('.clear-checkbox').forEach(function (cb) {
        cb.addEventListener('change', updateClearSelectedState);
      });
    }

    // ---------------- Sorting ----------------
    function initSorting() {
      document.querySelectorAll('#admin-task-table thead th.sortable').forEach(function (th) {
        th.addEventListener('click', function () {
          const key = th.dataset.key;
          if (currentSort.key === key) {
            currentSort.dir = currentSort.dir === 'asc' ? 'desc' : 'asc';
          } else {
            currentSort.key = key;
            currentSort.dir = 'asc';
          }
          document.querySelectorAll('#admin-task-table thead th.sortable .sort-indicator')
            .forEach(function (si) { si.textContent = ''; });
          const indicator = th.querySelector('.sort-indicator');
          if (indicator) {
            indicator.textContent = currentSort.dir === 'asc' ? '▲' : '▼';
          }
          render();
        });
      });
    }

    // ---------------- Event Bindings ----------------
    if (searchInput) {
      searchInput.addEventListener('input', render);
    }

    if (searchType) {
      searchType.addEventListener('change', render);
    }

    if (clearSearchBtn) {
      clearSearchBtn.addEventListener('click', function () {
        searchInput.value = '';
        searchType.value = 'all';
        render();
      });
    }

    if (refreshBtn) {
      refreshBtn.addEventListener('click', fetchTasks);
    }

    if (clearSelectedBtn) {
      clearSelectedBtn.addEventListener('click', clearSelected);
    }

    // ---------------- Init ----------------
    initSorting();
    fetchTasks();
  }
})();
