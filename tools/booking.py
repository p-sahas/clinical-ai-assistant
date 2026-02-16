"""
Booking lookup tool for patient appointments.

Looks up patient bookings from data/bookings.json by name, phone, or email.
Used by the multi-agent system when the user asks about "my appointment" or
identifies themselves (e.g. "I'm Sahas Induwara").
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def _resolve_bookings_path() -> Path:
    """Resolve path to data/bookings.json (works from project root or scratch_demos)."""
    cwd = Path.cwd()
    if (cwd / "data" / "bookings.json").exists():
        return cwd / "data" / "bookings.json"
    parent = cwd.parent
    if (parent / "data" / "bookings.json").exists():
        return parent / "data" / "bookings.json"
    return cwd / "data" / "bookings.json"


class BookingLookupTool:
    """
    Look up patient bookings by name, phone, or email.
    """

    def __init__(self, bookings_path: Optional[Path] = None):
        self.bookings_path = bookings_path or _resolve_bookings_path()
        self._data: Optional[Dict[str, Any]] = None

    def _load(self) -> Dict[str, Any]:
        if self._data is None:
            if not self.bookings_path.exists():
                self._data = {"patient_bookings": []}
                return self._data
            with open(self.bookings_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def lookup(
        self,
        patient_name: Optional[str] = None,
        patient_phone: Optional[str] = None,
        patient_email: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find bookings matching any of the given identifiers.
        Matching is case-insensitive; name supports partial match (e.g. "John" matches "John Silva").

        Args:
            patient_name: Full or partial patient name
            patient_phone: Phone number (full or partial)
            patient_email: Email address (full or partial)

        Returns:
            List of booking dicts (empty if none match)
        """
        data = self._load()
        bookings = data.get("patient_bookings", [])
        if not bookings:
            return []

        def norm(s: str) -> str:
            return (s or "").strip().lower()

        name = norm(patient_name or "")
        phone = norm(patient_phone or "")
        email = norm(patient_email or "")

        if not name and not phone and not email:
            return []

        matches = []
        for b in bookings:
            b_name = norm(b.get("patient_name", ""))
            b_phone = norm(b.get("patient_phone", ""))
            b_email = norm(b.get("patient_email", ""))

            if name and (name in b_name or b_name in name):
                matches.append(b)
                continue
            if phone and (phone in b_phone or b_phone in phone):
                matches.append(b)
                continue
            if email and (email in b_email or b_email in email):
                matches.append(b)
                continue

        # Deduplicate by booking_id (same booking might match multiple fields)
        seen = set()
        unique = []
        for m in matches:
            bid = m.get("booking_id")
            if bid and bid not in seen:
                seen.add(bid)
                unique.append(m)
            elif not bid:
                unique.append(m)
        return unique

    def format_bookings_for_agent(self, bookings: List[Dict[str, Any]]) -> str:
        """Format booking list as text for the Judge / agent consumption."""
        if not bookings:
            return "No matching bookings found."
        lines = []
        for i, b in enumerate(bookings, 1):
            lines.append(
                f"Booking {i}: {b.get('booking_id', 'N/A')} | "
                f"Patient: {b.get('patient_name')} | "
                f"Doctor: {b.get('doctor')} | "
                f"Specialty: {b.get('specialty')} | "
                f"Reason: {b.get('reason')} | "
                f"Date: {b.get('date')} at {b.get('time')} | "
                f"Hospital: {b.get('hospital')} | "
                f"Location: {b.get('location')} | "
                f"Status: {b.get('status')} | "
                f"Notes: {b.get('notes', 'None')}"
            )
        return "\n".join(lines)


def get_booking_tool(bookings_path: Optional[Path] = None) -> BookingLookupTool:
    """Return a BookingLookupTool instance."""
    return BookingLookupTool(bookings_path=bookings_path)
