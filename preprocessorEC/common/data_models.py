"""Data models for preprocessorEC.

Historically this repo stored "data models" as dictionaries holding DDL.
We keep that pattern for now, but also introduce small Python classes
that act as read-only models (not full ORM yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional


activeContracts = {
	"name": "PREPR.vw_ActiveContracts",
	"sql": (
		"CREATE OR ALTER VIEW PREPR.vw_ActiveContracts AS\n"
		"SELECT DISTINCT CONTRACT_NUMBER, CONTRACT_DESCRIPTION, MANUFACTURER_NAME, LAST_UPDATE_DATE\n"
		"FROM [DM_MONTYNT\\dli2].ccx_dump_validation_stg"
	),
}



def _format_ny_date(value: Any) -> Optional[str]:
	"""Return YYYY-MM-DD formatted in America/New_York.

	If the DB returns a date, it's already a calendar day and we keep it.
	If it returns a datetime, we interpret/convert it to New York time.
	"""
	if value is None:
		return None

	try:
		from zoneinfo import ZoneInfo  # py3.9+
		ny_tz = ZoneInfo('America/New_York')
	except Exception:  # pragma: no cover
		import pytz
		ny_tz = pytz.timezone('America/New_York')

	if isinstance(value, date) and not isinstance(value, datetime):
		return value.strftime('%Y-%m-%d')

	if isinstance(value, datetime):
		dt = value
		# If tz-naive, assume it's already NY local time.
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=ny_tz)
		return dt.astimezone(ny_tz).date().strftime('%Y-%m-%d')

	text = str(value).strip()
	if not text:
		return None

	# Best-effort parse for common DB string formats.
	for parser in (
		lambda s: datetime.fromisoformat(s),
		lambda s: datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S'),
		lambda s: datetime.strptime(s[:10], '%Y-%m-%d'),
	):
		try:
			dt = parser(text)
			if isinstance(dt, datetime) and dt.tzinfo is None:
				dt = dt.replace(tzinfo=ny_tz)
			return dt.astimezone(ny_tz).date().strftime('%Y-%m-%d')
		except Exception:
			continue

	# Fall back to first 10 chars when it looks like YYYY-MM-DD...
	return text[:10]


@dataclass(frozen=True)
class ActiveContract:
	"""Read-only row model for PREPR.vw_ActiveContracts."""

	contract_number: str
	manufacturer_name: Optional[str] = None
	contract_description: Optional[str] = None
	last_update_date: Optional[str] = None

	@staticmethod
	def search(conn: Any, query: str, limit: int = 20) -> list['ActiveContract']:
		"""Find active contracts whose number contains the query (case-insensitive)."""
		q = (query or '').strip()
		if not q:
			return []

		limit = int(limit) if limit else 20
		limit = max(1, min(limit, 50))

		cursor = conn.cursor()
		# SQL Server: use UPPER() + LIKE for case-insensitive matches across collations.
		sql = (
			f"SELECT TOP ({limit}) CONTRACT_NUMBER, CONTRACT_DESCRIPTION, MANUFACTURER_NAME, LAST_UPDATE_DATE "
			"FROM PREPR.vw_ActiveContracts "
			"WHERE UPPER(CONTRACT_NUMBER) LIKE ? "
			"ORDER BY CONTRACT_NUMBER"
		)
		cursor.execute(sql, (f"%{q.upper()}%",))
		rows = cursor.fetchall() or []
		items: list[ActiveContract] = []
		for row in rows:
			contract_number = (row[0] or '').strip()
			if not contract_number:
				continue
			items.append(
				ActiveContract(
					contract_number=contract_number,
					contract_description=(row[1] or '').strip() or None,
					manufacturer_name=(row[2] or '').strip() or None,
					last_update_date=_format_ny_date(row[1]),
				)
			)
		return items


__all__ = ["ActiveContract", "activeContracts"]

