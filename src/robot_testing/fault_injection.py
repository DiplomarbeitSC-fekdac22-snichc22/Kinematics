from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from state_machine.pick_and_place import MotionCommand, MotionCommandSink


class FaultPhase(str, Enum):
    """Point at which a configured fault is raised"""
    BEFORE_SEND = "before_send"
    AFTER_SEND = "after_send"


@dataclass(frozen=True)
class FaultRule:
    """Select one or more motion-sink calls on which to inject a fault.

    When both 'call_number' and 'command_name' are supplied, both must
    match. Call numbers are one-based. A non-repeating rule is consumed after
    its first match.
    """
    call_number: int | None = None
    command_name: str | None = None
    phase: FaultPhase = FaultPhase.BEFORE_SEND
    message: str = "Injected motion-sink fault"
    repeat: bool = False

    def __post_init__(self) -> None:
        if self.call_number is None and self.command_name is None:
            raise ValueError(
                "FaultRule requires call_number, command_name, or both"
            )
        if self.call_number is not None and self.call_number < 1:
            raise ValueError("FaultRule call_number must be at least 1")
        if self.command_name == "":
            raise ValueError("FaultRule command_name cannot be empty")

    def matches(self, call_number: int, command: MotionCommand) -> bool:
        if self.call_number is not None and call_number != self.call_number:
            return False
        if self.command_name is not None and command.name != self.command_name:
            return False
        return True


@dataclass(frozen=True)
class FaultOccurrence:
    """Trigger of a FaultRule"""
    call_number: int
    command_name: str
    phase: FaultPhase
    message: str


class InjectedMotionFault(RuntimeError):
    """Raised by FaultInjectingMotionSink for an injected failure."""
    def __init__(self, occurrence: FaultOccurrence) -> None:
        self.occurrence = occurrence
        super().__init__(
            f"Injected motion fault {occurrence.phase.value} at call "
            f"{occurrence.call_number} for command "
            f"{occurrence.command_name!r}: {occurrence.message}"
        )


class RecordingMotionSink:
    """Target of the fault injector"""
    def __init__(self) -> None:
        self.commands: list[MotionCommand] = []

    def send(self, command: MotionCommand) -> None:
        self.commands.append(command)


class FaultInjectingMotionSink:
    """Wrap another sink and inject deterministic before/after-send faults.

    'BEFORE_SEND' models a command that is known not to have reached the
    wrapped sink. 'AFTER_SEND' models ambiguous delivery: the wrapped sink
    accepted the command, but the caller still receives an exception.
    """
    def __init__(
        self,
        wrapped_sink: MotionCommandSink,
        rules: Iterable[FaultRule],
    ) -> None:
        self.wrapped_sink = wrapped_sink
        self.rules = tuple(rules)
        if not self.rules:
            raise ValueError("At least one FaultRule is required")

        self.attempted_commands: list[MotionCommand] = []
        self.delivered_commands: list[MotionCommand] = []
        self.faults: list[FaultOccurrence] = []

        self._call_count = 0
        self._consumed_rule_indexes: set[int] = set()

    @property
    def call_count(self) -> int:
        return self._call_count

    def send(self, command: MotionCommand) -> None:
        self._call_count += 1
        call_number = self._call_count
        self.attempted_commands.append(command)

        matched = self._matching_rule(call_number, command)
        if matched is not None:
            rule_index, rule = matched
            if rule.phase is FaultPhase.BEFORE_SEND:
                self._raise_fault(rule_index, rule, call_number, command)

        self.wrapped_sink.send(command)
        self.delivered_commands.append(command)

        if matched is not None:
            rule_index, rule = matched
            if rule.phase is FaultPhase.AFTER_SEND:
                self._raise_fault(rule_index, rule, call_number, command)

    def _matching_rule(
        self,
        call_number: int,
        command: MotionCommand,
    ) -> tuple[int, FaultRule] | None:
        for index, rule in enumerate(self.rules):
            if index in self._consumed_rule_indexes:
                continue
            if rule.matches(call_number, command):
                return index, rule
        return None

    def _raise_fault(
        self,
        rule_index: int,
        rule: FaultRule,
        call_number: int,
        command: MotionCommand,
    ) -> None:
        if not rule.repeat:
            self._consumed_rule_indexes.add(rule_index)

        occurrence = FaultOccurrence(
            call_number=call_number,
            command_name=command.name,
            phase=rule.phase,
            message=rule.message,
        )
        self.faults.append(occurrence)
        raise InjectedMotionFault(occurrence)
