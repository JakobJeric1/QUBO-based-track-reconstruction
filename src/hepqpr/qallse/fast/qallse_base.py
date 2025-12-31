import itertools
import logging
import sys
import time
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numba import njit
import pandas as pd
from ..other.stdout_redirect import capture_stdout

from .data_structures import *
from .data_wrapper import DataWrapper
from .utils import tracks_to_xplets_numba


# ==========================
# PERF: numba helpers (no fallbacks)
# ==========================

@njit(cache=True)
def _emit_pairs_one_conflict_list(conflict_doublets, dblet_triplets, out_pairs, start_idx):
    n = conflict_doublets.shape[0]
    for a in range(n):
        d1 = conflict_doublets[a]
        tlist1 = dblet_triplets[d1]
        for b in range(a + 1, n):
            d2 = conflict_doublets[b]
            tlist2 = dblet_triplets[d2]
            for i in range(tlist1.shape[0]):
                t1 = tlist1[i]
                for j in range(tlist2.shape[0]):
                    t2 = tlist2[j]
                    if t1 == t2:
                        continue
                    out_pairs[start_idx, 0] = t1
                    out_pairs[start_idx, 1] = t2
                    start_idx += 1
    return start_idx


@njit(cache=True)
def _build_conflict_pairs(all_hit_conflicts, dblet_triplets, max_pairs):
    out = np.empty((max_pairs, 2), dtype=np.int64)
    idx = 0
    for k in range(len(all_hit_conflicts)):
        conflicts = all_hit_conflicts[k]
        if conflicts.shape[0] < 2:
            continue
        idx = _emit_pairs_one_conflict_list(conflicts, dblet_triplets, out, idx)
        if idx >= max_pairs:
            return out[:idx]
    return out[:idx]


class ConfigBase(ABC):
    """Encapsulate parameters for a model. The parameters can be defined as class attributes."""

    def as_dict(self):
        """Return the current configuration as a dictionary."""
        return dict([(k, getattr(self, k)) for k in dir(self)
                     if not k.startswith('_') and not k in ['as_dict', 'update']])

    def update(self, logger=None, **kwargs):
        """Update this configuration, doing type conversions if necessary."""
        for k, v in kwargs.items():
            try:
                if hasattr(self, k):
                    typ = type(getattr(self, k))
                    if type(v) != typ: v = typ(v)
                    setattr(self, k, v)
                else:
                    logger.warning(f'got an unknown config parameter {k}={v}')
            except Exception:
                if logger is not None:
                    logger.warning(f'ignored config {k}={v}, wrong type (should be of {typ})')


# ============================================================

class QallseBase(ABC):
    """Abstract base class of a Qallse model. Handles everything except the hard cuts and the qubo weights computations."""

    def __init__(self, dataw: DataWrapper, **config):
        """Initialise a model with the given dataset. All other parameters will override the default configuration."""

        self.logger = logging.getLogger(self.__module__)
        self.dataw = dataw

        self.config = self._get_base_config()
        self.config.update(logger=self.logger, **config)
        self.logger.debug(f'using config:')
        for (k, v) in self.config.as_dict().items(): self.logger.debug(f'    {k}: {v}')

        #: All hits generated
        self.hits: Dict[str, Hit] = {}
        #: All doublets generated
        self.doublets: List[Doublet] = []
        #: All triplets generated
        self.triplets: List[Triplet] = []
        #: All quadruplets generated
        self.quadruplets: List[Quadruplet] = []

        #: Triplets that will figure in the QUBO as variables
        self.qubo_triplets: Set[Triplet] = set()
        #: Doublets used by at least one triplet included in the QUBO
        self.qubo_doublets: Set[Doublet] = set()
        #: Hits used by at least one Xplet in the QUBO
        self.qubo_hits: Dict[str, Hit] = {}

        for row in self.dataw.hits.iterrows():
            h = Hit(**dict(row[1]))
            self.hits[h.hit_id] = h

    @abstractmethod
    def _get_base_config(self):
        # [ABSTRACT] Return an instance of a subclass of `ConfigBase` holding all model parameters
        pass

    def build_model(self, doublets: Union[pd.DataFrame, List, np.array]):
        """
        Do the preprocessing, i.e. prepare everything so the QUBO can be generated.
        This includes creating the structures (hits, doublets, triplets, quadruplets) and computing the weights.

        :param doublets: the input doublets
        :return: self (for chaining)
        """
        start_time = time.process_time()
        self.hard_cuts_stats = self.hard_cuts_stats[:1]

        initial_doublets = doublets.values if isinstance(doublets, pd.DataFrame) else doublets

        self._create_doublets(initial_doublets)
        self._create_triplets()
        self._create_quadruplets()

        end_time = time.process_time() - start_time

        self.logger.info(
            f'Model built in {end_time:.2f}s. '
            f'doublets: {len(self.doublets)}/{len(self.qubo_doublets)}, '
            f'triplets: {len(self.triplets)}/{len(self.qubo_triplets)}, '
            f'quadruplets: {len(self.quadruplets)}')

        return self

    def sample_qubo(self, Q: TQubo = None, return_time=False, logfile: str = None, seed: int=None, **qbsolv_params) -> Union[
        object, Tuple[object, float]]:
        """
        Submit a QUBO to (see `qbsolv <https://github.com/dwavesystems/qbsolv>`_).

        :param Q: the QUBO. If not defined, :py:meth:~`to_qubo` will be called.
        :param return_time: if set, also return the execution time (in seconds)
        :param qbsolv_params: parameters to pass to qbsolv's `sample_qubo` method
        :param logfile: path to a file. if set, all qbsolv output will be redirected to this file
        :param seed: the random seed for qbsolv. **MUST** be a 32-bits integer with entropy !
        :return: a dimod response or a tuple (dimod response, exec_time)
         (see `dimod.Response <https://docs.ocean.dwavesys.com/projects/dimod/en/latest/reference/response.html>`_)
        """
        if Q is None: Q = self.to_qubo()
        if seed is None:
            import random
            seed = random.randint(0, 1<<31)

        # run qbsolv
        start_time = time.process_time()
        try:
            with capture_stdout(logfile):
                response = QBSolv().sample_qubo(Q, seed=seed, **qbsolv_params)
        except: # fails if called from ipython notebook...
            response = QBSolv().sample_qubo(Q, seed=seed, **qbsolv_params)

        exec_time = time.process_time() - start_time

        self.logger.info(f'QUBO of size {len(Q)} sampled in {exec_time:.2f}s (seed {seed}).')

        return (response, exec_time) if return_time else response

    @classmethod
    def process_sample(self, sample: TDimodSample) -> List[TXplet]:
        """
        Convert a QUBO solution into a set of doublets.
        The sample needs to behave like a dictionary, but can also be an instance of dimod.SampleView.

        :param sample: the QUBO response to process
        :return: the list of final doublets
        """
        final_triplets = [np.array(Triplet.name_to_hit_ids(k), dtype=np.int64) for k, v in sample.items() if v == 1]
        final_doublets = tracks_to_xplets_numba(final_triplets, x=2)
        return np.unique(final_doublets, axis=0).tolist()


    # ---------------------------------------------

    def _create_doublets(self, initial_doublets):
        # Generate Doublet structures from the initial doublets, calling _is_invalid_doublet to apply early cuts
        doublets = []
        for (start_id, end_id) in initial_doublets:
            start, end = self.hits[start_id], self.hits[end_id]
            d = Doublet(start, end)
            if not self._is_invalid_doublet(d):
                start.outer.append(d)
                end.inner.append(d)
                doublets.append(d)

        self.logger.info(f'created {len(doublets)} doublets.')
        self.doublets = doublets

    @abstractmethod
    def _is_invalid_doublet(self, dblet: Doublet) -> bool:
        # [ABSTRACT] Apply early cuts on doublets, return True if the doublet should be discarded.
        pass

    def _create_triplets(self):
        # Generate Triplet structures from Doublets, calling _is_invalid_triplet to apply early cuts
        triplets = []
        for d1 in self.doublets:
            for d2 in d1.h2.outer:
                t = Triplet(d1, d2)
                if not self._is_invalid_triplet(t):
                    d1.outer.append(t)
                    d2.inner.append(t)
                    triplets.append(t)
        self.logger.info(f'created {len(triplets)} triplets.')
        self.triplets = triplets

    @abstractmethod
    def _is_invalid_triplet(self, tplet: Triplet) -> bool:
        # [ABSTRACT] Apply early cuts on triplets, return True if the triplet should be discarded.
        pass

    def _create_quadruplets(self, register_qubo=True):
        # Generate Quadruplet structures from Triplets, calling _is_invalid_quadruplet to apply early cuts
        # If register_qubo is True, the model will be "cleaned" at the same time, saving one iteration
        # Here, cleaning means registering the quadruplet as used in the qubo (see _register_qubo_quadruplet).
        # Set it to False if you plan to do another pass of filtering on quadruplet (in this case, it is your
        # responsibility to call _register_qubo_quadruplet) or if you want to include all the generated triplet
        # in the QUBO (in this case, you have to set the `qubo_*` structures properly)
        quadruplets = []
        for t1 in self.triplets:
            for t2 in t1.d2.outer:
                qplet = Quadruplet(t1, t2)
                if not self._is_invalid_quadruplet(qplet):
                    t1.outer.append(qplet)
                    t2.inner.append(qplet)
                    qplet.strength = self._compute_strength(qplet)
                    quadruplets.append(qplet)
                    if register_qubo:
                        self._register_qubo_quadruplet(qplet)

        self.logger.info(f'created {len(quadruplets)} quadruplets.')
        self.quadruplets = quadruplets

    @abstractmethod
    def _is_invalid_quadruplet(self, qplet: Quadruplet) -> bool:
        # [ABSTRACT] Apply early cuts on quadruplets, return True if the quadruplet should be discarded.
        pass

    def _register_qubo_quadruplet(self, qplet: Quadruplet):
        # Mark all triplets, doublets and hits used by this quadruplet as "kept", so that they will be used
        # when constructing the QUBO.
        for d in qplet.doublets():
            d.h1.outer_kept.add(d)
            d.h2.inner_kept.add(d)
        for t in [qplet.t1, qplet.t2]:
            t.d1.outer_kept.add(t)
            t.d2.inner_kept.add(t)

        self.qubo_doublets.update(qplet.doublets())
        self.qubo_hits.update(zip(qplet.hit_ids(), qplet.hits))
        self.qubo_triplets.update([qplet.t1, qplet.t2])

    @abstractmethod
    def _compute_weight(self, tplet: Triplet) -> float:
        # [ABSTRACT] Return the bias weight (linear) that should be associated to this triplet in the QUBO
        pass

    @abstractmethod
    def _compute_strength(self, qplet: Quadruplet) -> float:
        # [ABSTRACT] Return the coupling strength associated to the coupler between `qplet.t1` and `qplet.t2` in the
        # QUBO. This will be called for each quadruplet generated is should in theory be negative to *tie* `t1`
        # and `t2` together (as opposed to _compute_conflict_strength)
        pass

    @abstractmethod
    def _compute_conflict_strength(self, t1: Triplet, t2: Triplet):
        # [ABSTRACT] Return the coupling strength associated to the coupler between `t1` and `t2` in the
        # QUBO. If this method is called, it means that `t1` and `t2` are in conflict: they are both using
        # one or more hits in a conflicting manner and should not both figure in the solution. Hence, we expect
        # the return value to be positive.
        pass

    # ---------------------------------------------

    def to_qubo(self, return_stats=False) -> Union[TQubo, Tuple[TQubo, Tuple[int, int, int]]]:
        """
        Generate the QUBO. Attention: ensure that :py:meth:`build_model` has been called previously.
        :param return_stats: if set, also return the number of variables and coulpers.
        :return: either the QUBO, or a tuple (QUBO, (n_vars, n_incl_couplers, n_excl_couplers))
        """

        Q = {}
        hits, doublets, triplets = self.qubo_hits, self.qubo_doublets, self.qubo_triplets
        quadruplets = self.quadruplets

        start_time = time.process_time()

        # --- PERF: cache string names once (str() is expensive in hot loops)
        for t in triplets:
            if not hasattr(t, "_name"):
                t._name = str(t)
        for q in quadruplets:
            if not hasattr(q.t1, "_name"):
                q.t1._name = str(q.t1)
            if not hasattr(q.t2, "_name"):
                q.t2._name = str(q.t2)

        # 1: qbits with their weight (doublets with a common weight)
        for t in triplets:
            t.weight = self._compute_weight(t)
            name = t._name
            Q[(name, name)] = t.weight
        n_vars = len(Q)

        # 2a: exclusion couplers (no two triplets can share the same doublet)
        # PERF: pure-python fast path (no itertools.combinations, no chain, no Q membership checks on strings)

        triplets_list = list(triplets)
        triplet_to_idx = {t: i for i, t in enumerate(triplets_list)}
        compute_conflict = self._compute_conflict_strength  # local binding

        seen = set()  # store (min_idx, max_idx) once

        # --- PERF: cache "triplets using each doublet" once (avoid list allocations in hot loop)
        dblet_ts = {}
        for d in doublets:
            if d.inner_kept or d.outer_kept:
                lst = []
                lst.extend(d.inner_kept)
                lst.extend(d.outer_kept)
                dblet_ts[d] = lst


        for hit in hits.values():
            # we need to process both inner_kept and outer_kept like original code
            for conflicts in (hit.inner_kept, hit.outer_kept):
                if len(conflicts) < 2:
                    continue

                # list() once (indexable, faster than set iteration repeatedly)
                conf_list = list(conflicts)
                m = len(conf_list)

                for i in range(m - 1):
                    d1 = conf_list[i]
                    ts1 = dblet_ts.get(d1)
                    if not ts1:
                        continue


                    for j in range(i + 1, m):
                        d2 = conf_list[j]
                        ts2 = dblet_ts.get(d2)
                        if not ts2:
                            continue


                        # cross product of triplets using d1 and d2
                        for t1 in ts1:
                            i1 = triplet_to_idx[t1]
                            for t2 in ts2:
                                if t1 is t2:
                                    continue
                                i2 = triplet_to_idx[t2]


                                if i1 < i2:
                                    edge = (i1, i2)
                                else:
                                    edge = (i2, i1)

                                if edge in seen:
                                    continue
                                seen.add(edge)

                                # only now touch strings
                                Q[(triplets_list[edge[0]]._name, triplets_list[edge[1]]._name)] = compute_conflict(t1, t2)


        n_excl_couplers = len(Q) - n_vars

        # 2b: inclusion couplers (consecutive doublets with a good triplet)
        for q in quadruplets:
            Q[(q.t1._name, q.t2._name)] = q.strength

        n_incl_couplers = len(Q) - (n_vars + n_excl_couplers)
        exec_time = time.process_time() - start_time

        self.logger.info(
            f'Qubo generated in {exec_time:.2f}s. Size: {len(Q)}. Vars: {n_vars}, '
            f'excl. couplers: {n_excl_couplers}, incl. couplers: {n_incl_couplers}'
        )
        if return_stats:
            return Q, (n_vars, n_incl_couplers, n_excl_couplers)
        else:
            return Q

