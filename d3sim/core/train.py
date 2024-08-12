"""tiny generic train engine, event emitter based, support base train event and custom period event.
"""

from contextlib import nullcontext
import dataclasses
import traceback
from typing import Any, Callable, ContextManager, Dict, Generator, Generic, Iterable, List, Optional, Tuple, TypeVar, Union
import pyee

import enum


class TrainEventType(enum.Enum):
    BeginIteration = "BeginIteration"
    EndIteration = "EndIteration"
    BeginTrain = "BeginTrain"

    EndTrain = "EndTrain"

    BeginEpoch = "BeginEpoch"

    EndEpoch = "EndEpoch"
    Exception = "Exception"


T = TypeVar("T")
T_co = TypeVar("T_co")


@dataclasses.dataclass
class TrainEvent:
    type: str
    event_index: int
    cur_step: int
    cur_epoch: int = -1
    total_step: int = -1
    total_epoch: int = -1
    total_step_per_epoch: int = -1


@dataclasses.dataclass
class StepEvent(Generic[T, T_co]):
    data: T
    cur_step: int
    cur_epoch: int = -1
    step_ctx_value: Optional[T_co] = None


@dataclasses.dataclass
class PeriodEventDesp:
    interval: int
    times: int


class PeriodEventEngine(Generic[T]):

    def __init__(self) -> None:
        self._event_desps: Dict[Tuple[T, bool], PeriodEventDesp] = {}

    def __contains__(self, event_tuple: Tuple[T, bool]):
        return event_tuple in self._event_desps

    def __getitem__(self, event_tuple: Tuple[T, bool]):
        return self._event_desps[event_tuple]

    def register_event(self,
                       event: T,
                       interval: int,
                       times: int = -1,
                       before: bool = False):
        assert interval > 0
        assert event not in self._event_desps
        self._event_desps[(event, before)] = PeriodEventDesp(interval, times)

    def query_events(self, cur_step: int, before: bool = False):
        res: List[Tuple[T, int]] = []
        for k, v in self._event_desps.items():
            ev_type = k[0]
            is_before = k[1]
            if is_before != before:
                continue
            if (cur_step + 1) % v.interval == 0:
                event_idx = (cur_step + 1) // v.interval
                if v.times <= 0 or (v.times > 0 and event_idx <= v.times):
                    res.append((ev_type, event_idx))
        return res


class EpochGenerator(Generic[T_co]):

    def __init__(self,
                 cur_epoch: int,
                 cur_step: int,
                 total_epoch: int,
                 event_emitter: pyee.EventEmitter,
                 period_event_engine: PeriodEventEngine[str],
                 total_step_per_epoch: int = -1,
                 total_step: int = -1,
                 step_ctx_creator: Optional[Callable[[],
                                                     ContextManager[T_co]]] = None):
        self.event_emitter = event_emitter
        self.period_event_engine = period_event_engine
        self.cur_step = cur_step
        self.cur_epoch = cur_epoch
        self.total_epoch = total_epoch
        if total_step_per_epoch != -1:
            if total_step == -1:
                total_step = total_step_per_epoch * total_epoch
            assert total_step_per_epoch * total_epoch == total_step
        self.total_step_per_epoch = total_step_per_epoch
        self.total_step = total_step
        self.step_ctx_creator = step_ctx_creator

    def epoch_step_generator(
            self,
            step_generator: Iterable[T],
            cur_step: int = -1) -> Generator[StepEvent[T, T_co], None, None]:
        if cur_step != -1:
            self.cur_step = cur_step
        total_step_int = cur_step
        base_event = TrainEvent(TrainEventType.BeginIteration.value,
                                self.cur_step,
                                self.cur_step,
                                self.cur_epoch,
                                total_epoch=self.total_epoch,
                                total_step=self.total_step,
                                total_step_per_epoch=self.total_step_per_epoch)
        for data in step_generator:
            step_ctx = nullcontext()
            if self.step_ctx_creator is not None:
                step_ctx = self.step_ctx_creator()
            with step_ctx as ctx_value:
                events = self.period_event_engine.query_events(
                    self.cur_step, True)
                for ev_type, ev_index in events:
                    self.event_emitter.emit(
                        ev_type,
                        dataclasses.replace(base_event,
                                            type=ev_type,
                                            event_index=ev_index,
                                            cur_step=self.cur_step))

                self.event_emitter.emit(
                    TrainEventType.BeginIteration.value,
                    dataclasses.replace(
                        base_event,
                        type=TrainEventType.BeginIteration.value,
                        cur_step=self.cur_step,
                        event_index=self.cur_step))
                try:
                    ev = StepEvent(
                        data, self.cur_step, self.cur_epoch,
                        None if self.step_ctx_creator is None else ctx_value)
                    yield ev
                except:
                    self.event_emitter.emit(
                        TrainEventType.Exception.value,
                        dataclasses.replace(
                            base_event,
                            type=TrainEventType.Exception.value,
                            cur_step=self.cur_step,
                            event_index=self.cur_step))
                    raise
                self.event_emitter.emit(
                    TrainEventType.EndIteration.value,
                    dataclasses.replace(base_event,
                                        type=TrainEventType.EndIteration.value,
                                        cur_step=self.cur_step,
                                        event_index=self.cur_step))
                events = self.period_event_engine.query_events(
                    self.cur_step, False)
                for ev_type, ev_index in events:
                    self.event_emitter.emit(
                        ev_type,
                        dataclasses.replace(base_event,
                                            type=ev_type,
                                            event_index=ev_index,
                                            cur_step=self.cur_step))
                self.cur_step += 1
                total_step_int += 1


class BasicTrainEngine:
    """basic engine that support epoch/iteration/endtrain event.
    stateless (except EpochGenerator).
    """
    TrainEventType = TrainEventType

    def __init__(self) -> None:
        self.event_emitter = pyee.EventEmitter()
        self.period_event_engine: PeriodEventEngine[str] = PeriodEventEngine()
        self.epoch_period_event_engine: PeriodEventEngine[
            str] = PeriodEventEngine()

    def register_base_event(self, event_type: TrainEventType,
                            handler: Callable[[TrainEvent], Any]):
        self.event_emitter.on(event_type.value, handler)

    def register_period_event(self,
                              event_key: str,
                              period: int,
                              handler: Callable[[TrainEvent], Any],
                              times: int = -1,
                              before: bool = False):
        """for period event, you need to specify a unique key to indentify your event handler.
        """
        if (event_key, before) not in self.period_event_engine:
            self.period_event_engine.register_event(event_key, period, times,
                                                    before)
        else:
            assert self.period_event_engine[(event_key, before)] == period
        self.event_emitter.on(event_key, handler)

    def register_epoch_period_event(self,
                                    event_key: str,
                                    period: int,
                                    handler: Callable[[TrainEvent], Any],
                                    times: int = -1,
                                    before: bool = False):
        if (event_key, before) not in self.epoch_period_event_engine:
            self.epoch_period_event_engine.register_event(
                event_key, period, times)
        else:
            assert self.epoch_period_event_engine[(event_key,
                                                   before)] == period
        self.event_emitter.on(event_key, handler)

    def train_step_generator(
        self,
        start_step: int,
        step_generator: Iterable[T],
        total_step: int = -1,
        step_ctx_creator: Optional[Callable[[], ContextManager[T_co]]] = None
    ) -> Generator[StepEvent[T, T_co], None, None]:

        base_event = TrainEvent(TrainEventType.BeginIteration.value,
                                start_step,
                                start_step,
                                -1,
                                total_epoch=-1,
                                total_step=total_step)

        self.event_emitter.emit(
            TrainEventType.BeginTrain.value,
            TrainEvent(TrainEventType.BeginTrain.value, start_step,
                       start_step))
        total_step_int = start_step
        cur_step = start_step
        for data in step_generator:
            step_ctx = nullcontext()
            if step_ctx_creator is not None:
                step_ctx = step_ctx_creator()
            with step_ctx as ctx_value:
                events = self.period_event_engine.query_events(cur_step, True)
                for ev_type, ev_index in events:
                    self.event_emitter.emit(
                        ev_type,
                        dataclasses.replace(base_event,
                                            type=ev_type,
                                            cur_step=cur_step,
                                            event_index=ev_index))

                self.event_emitter.emit(
                    TrainEventType.BeginIteration.value,
                    dataclasses.replace(
                        base_event,
                        type=TrainEventType.BeginIteration.value,
                        cur_step=cur_step,
                        event_index=cur_step))
                try:
                    ev = StepEvent(data,
                                    cur_step,
                                    step_ctx_value=None
                                    if step_ctx_creator is None else ctx_value)
                    yield ev
                except:
                    self.event_emitter.emit(
                        TrainEventType.Exception.value,
                        dataclasses.replace(
                            base_event,
                            type=TrainEventType.Exception.value,
                            cur_step=cur_step,
                            event_index=cur_step))
                    raise
                self.event_emitter.emit(
                    TrainEventType.EndIteration.value,
                    dataclasses.replace(base_event,
                                        type=TrainEventType.EndIteration.value,
                                        cur_step=cur_step,
                                        event_index=cur_step))

                events = self.period_event_engine.query_events(cur_step)
                for ev_type, ev_index in events:
                    self.event_emitter.emit(
                        ev_type,
                        dataclasses.replace(base_event,
                                            type=ev_type,
                                            cur_step=cur_step,
                                            event_index=ev_index))

                cur_step += 1
                total_step_int += 1
        self.event_emitter.emit(
            TrainEventType.EndTrain.value,
            dataclasses.replace(base_event,
                                type=TrainEventType.EndTrain.value,
                                cur_step=total_step_int,
                                event_index=total_step_int))

    def train_epoch_generator(
        self,
        cur_step: int,
        start_epoch: int,
        total_epoch: int,
        total_step_per_epoch: int = -1,
        total_step: int = -1,
        step_ctx_creator: Optional[Callable[[], ContextManager[T_co]]] = None
    ) -> Generator[EpochGenerator[T_co], None, None]:
        if total_step_per_epoch != -1:
            if total_step == -1:
                total_step = total_step_per_epoch * total_epoch
            assert total_step_per_epoch * total_epoch == total_step
        base_event = TrainEvent(TrainEventType.BeginIteration.value,
                                cur_step,
                                cur_step,
                                start_epoch,
                                total_epoch=total_epoch,
                                total_step=total_step,
                                total_step_per_epoch=total_step_per_epoch)

        self.event_emitter.emit(
            TrainEventType.BeginTrain.value,
            dataclasses.replace(
                base_event,
                type=TrainEventType.BeginTrain.value,
                cur_step=cur_step,
                event_index=cur_step,
            ))
        for epoch in range(start_epoch, total_epoch):
            events = self.epoch_period_event_engine.query_events(epoch, True)
            for ev_type, ev_index in events:
                self.event_emitter.emit(
                    ev_type,
                    dataclasses.replace(
                        base_event,
                        type=ev_type,
                        cur_step=cur_step,
                        event_index=ev_index,
                        cur_epoch=epoch,
                    ))

            self.event_emitter.emit(
                TrainEventType.BeginEpoch.value,
                dataclasses.replace(
                    base_event,
                    type=TrainEventType.BeginEpoch.value,
                    cur_step=cur_step,
                    event_index=cur_step,
                    cur_epoch=epoch,
                ))
            epg = EpochGenerator(epoch, cur_step, total_epoch,
                                 self.event_emitter, self.period_event_engine,
                                 total_step_per_epoch, total_step,
                                 step_ctx_creator)
            yield epg
            cur_step = epg.cur_step
            self.event_emitter.emit(
                TrainEventType.EndEpoch.value,
                dataclasses.replace(
                    base_event,
                    type=TrainEventType.EndEpoch.value,
                    cur_step=cur_step,
                    event_index=cur_step,
                    cur_epoch=epoch,
                ))

            events = self.epoch_period_event_engine.query_events(epoch, False)
            for ev_type, ev_index in events:
                self.event_emitter.emit(
                    ev_type,
                    dataclasses.replace(
                        base_event,
                        type=ev_type,
                        cur_step=cur_step,
                        event_index=ev_index,
                        cur_epoch=epoch,
                    ))

        self.event_emitter.emit(
            TrainEventType.EndTrain.value,
            dataclasses.replace(
                base_event,
                type=TrainEventType.EndTrain.value,
                cur_step=cur_step,
                event_index=total_epoch,
                cur_epoch=total_epoch,
            ))
