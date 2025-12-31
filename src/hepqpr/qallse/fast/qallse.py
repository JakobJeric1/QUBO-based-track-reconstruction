import math
import numpy as np
import pandas as pd
from .data_structures import *
from .qallse_base import ConfigBase, QallseBase
from .utils import pd_read_csv_array


_PI = math.pi
_TWO_PI = 2.0 * math.pi

def angle_diff_fast(a1, a2):
    da = abs(a2 - a1)
    return da if da <= _PI else _TWO_PI - da

def menger_curv_fast(x0, y0, x1, y1, x2, y2):
    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = x2 - x0
    dy2 = y2 - y0
    twice_area = dx1 * dy2 - dy1 * dx2
    len0 = math.hypot(dx1, dy1)
    len1 = math.hypot(x2 - x1, y2 - y1)
    len2 = math.hypot(dx2, dy2)
    return 2.0 * twice_area / (len0 * len1 * len2)


class Config(ConfigBase):
    cheat = False

    # === Hard cut

    #: Doublets can miss at most (max_layer_span - 1) layers.
    #: Note that triplets and quadruplets also have this limitation, i.e.
    #: any xplet will at most miss (max_layer_span - 1) layers
    max_layer_span = 2

    #: Maximum radius of curvature for a triplet. The curvature is computed using
    #: the *Mengel curvature*.
    tplet_max_curv = 5E-3
    #: Maximum (absolute) difference between the angles in the R-Z plane of the two doublets forming
    #: the triplet. The angles are defined as arctan(dz/dr).
    tplet_max_drz = 0.2

    #: Maximum difference between the radius of curvature of the two triplets forming the quadruplet.
    qplet_max_dcurv = 5E-4
    #: Maximum strength of a quadruplet. This cut is really efficient, but the actual value depends
    #: highly on the strength function parameters (see below)
    qplet_max_strength = -0.2

    #: Linear bias weight associated to triplets in the QUBO.
    qubo_bias_weight = 0
    #: Quadratic coupling strength associated to two conflicting triplets in the QUBO.
    #: Set it to 1 (other things being equal) to avoid conflicts.
    qubo_conflict_strength = 1

    # === strength computation

    #: Factor of the numerator in the strength formula. Should be negative.
    num_multiplier = -1
    #: Ponderation between the curvature (X-Y plane) and the delta angle (R-Z plane) in the numerator.
    #: Should be a percentage (0 <= `xy_relative_strength` <= 1).
    xy_relative_strength = 0.5
    #: Exponent of the curvature (X-Y plane) in the strength formula. Should be >= 0.
    xy_power = 1
    #: Exponent of the delta angle (R-Z plane) in the strength formula. Should be >= 0.
    rz_power = 1
    #: Exponent of the "layer miss" in the strength formula (denominator). Should be >= 0.
    volayer_power = 2
    #: Clipping bounds of the strength. If defined, strength values outside those bounds will take the
    #: value of the bound.
    strength_bounds = None


class Config1GeV(Config):
    tplet_max_curv = 8E-4  # (vs 5E-3)
    tplet_max_drz = 0.1  # (vs0.2)
    qplet_max_dcurv = 1E-4  # (vs4E-4)


class Qallse(QallseBase):
    config: Config  # for proper autocompletion in PyCharm

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hard_cuts_stats = ['type,hid,reason,arg1,arg2']

    def _get_base_config(self):
        return Config1GeV() # TODO

    def get_build_stats(self) -> pd.DataFrame:
        """Return a dataframe, each row corresponding to a real xplet that has been dropped during preprocessing."""
        assert len(self.hard_cuts_stats) >= 1  # ensure it has headers
        return pd_read_csv_array(self.hard_cuts_stats)

    def build_model(self, *args, **kwargs):
        super().build_model(*args, **kwargs)
        # add stats information to the logs
        self.log_build_stats()

    def log_build_stats(self):
        """ Log information about real doublets/triplets/quadruplets dropped during model building"""
        stats = self.get_build_stats()
        if stats.shape[0] > 0:
            self.logger.info(f'Dropped {len(stats)} real structures during preprocessing')
            if len(stats) <= 10:
                self.logger.debug('\n' + stats.to_string())
            details = 'Dropped type:reason:count => '
            for (typ, reason), df in stats.groupby(['type', 'reason']):
                details += f'{typ}:{reason}:{len(df)} '
            self.logger.info(details)

    # --------------- early cuts

    def _is_invalid_doublet(self, dblet: Doublet) -> bool:
        # Apply hard cuts on doublets.
        # Currently, doublets are only dropped if they miss more than one layer.

        v1, v2 = dblet.h1.volayer, dblet.h2.volayer
        ret = v1 >= v2 or v2 > v1 + self.config.max_layer_span
        if ret and self.dataw.is_real_doublet(dblet.hit_ids()) == XpletType.REAL:
            self.hard_cuts_stats.append(f'dblet,{dblet},volayer,{v1},{v2}')
            return not self.config.cheat
        return ret

    def _create_triplets(self):
        """
        PERF: Fast triplet creation with early cuts BEFORE instantiating Triplet objects.
        Logic/results are identical to _is_invalid_triplet, but we avoid creating Triplet()
        for candidates that fail quickly.
        """
        triplets = []

        # local bindings (faster than repeated attribute lookups)
        max_layer_span = self.config.max_layer_span
        tplet_max_curv = self.config.tplet_max_curv
        tplet_max_drz = self.config.tplet_max_drz
        cheat = self.config.cheat
        hard_stats = self.hard_cuts_stats
        is_real_xplet = self.dataw.is_real_xplet


        # Generate Triplet structures from Doublets, with early cuts
        for d1 in self.doublets:
            # d1.h2.outer contains doublets starting where d1 ends
            for d2 in d1.h2.outer:

                h0 = d1.h1
                h1 = d1.h2  # == d2.h1
                h2 = d2.h2

                # ---- CUT 1: layer skips (same as in _is_invalid_triplet)
                volayer_skip = h2.volayer - h0.volayer
                if volayer_skip > max_layer_span + 1:
                    # only compute "real" if we need it for logging/cheat behavior
                    if is_real_xplet([h0.hit_id, h1.hit_id, h2.hit_id]) == XpletType.REAL:
                        name = f"{h0.hit_id}_{h1.hit_id}_{h2.hit_id}"
                        hard_stats.append(f"tplet,{name},volayer,{volayer_skip},")
                        if not cheat:
                            continue
                    else:
                        continue

                # ---- CUT 2: curvature (same as Triplet.curvature + _is_invalid_triplet)
                curv = menger_curv_fast(h0.x, h0.y, h1.x, h1.y, h2.x, h2.y)
                if abs(curv) > tplet_max_curv:
                    if is_real_xplet([h0.hit_id, h1.hit_id, h2.hit_id]) == XpletType.REAL:
                        name = f"{h0.hit_id}_{h1.hit_id}_{h2.hit_id}"
                        hard_stats.append(f"tplet,{name},curv,{curv},")
                        if not cheat:
                            continue
                    else:
                        continue

                # ---- CUT 3: drz (same as Triplet.drz + _is_invalid_triplet)
                drz = angle_diff_fast(d1.rz_angle, d2.rz_angle)
                if drz > tplet_max_drz:
                    if is_real_xplet([h0.hit_id, h1.hit_id, h2.hit_id]) == XpletType.REAL:
                        name = f"{h0.hit_id}_{h1.hit_id}_{h2.hit_id}"
                        hard_stats.append(f"tplet,{name},drz,{drz},")
                        if not cheat:
                            continue
                    else:
                        continue

                # If we are here: accepted -> create the real Triplet object and wire it in
                t = Triplet(d1, d2, curvature_val=curv, drz_val=drz)
                d1.outer.append(t)
                d2.inner.append(t)
                triplets.append(t)

        self.logger.info(f"created {len(triplets)} triplets.")
        self.triplets = triplets


    def _create_quadruplets(self, register_qubo=True):
        """
        PERF: Fast quadruplet creation with early cuts BEFORE instantiating Quadruplet objects.
        Logic/results should match _is_invalid_quadruplet, but we avoid creating Quadruplet()
        for candidates that fail quickly.
        """
        quadruplets = []

        # local bindings (faster than repeated attribute lookups)
        qplet_max_dcurv = self.config.qplet_max_dcurv
        qplet_max_strength = self.config.qplet_max_strength
        cheat = self.config.cheat
        hard_stats = self.hard_cuts_stats
        is_real_xplet = self.dataw.is_real_xplet

        # iterate triplet chains
        for t1 in self.triplets:
            # t1.d2.outer are triplets that can follow t1 (share d2 == next.d1)
            for t2 in t1.d2.outer:
                # ---- EARLY CUT 1: delta curvature
                dcurv = abs(t1.curvature - t2.curvature)
                if dcurv > qplet_max_dcurv:
                    # log only if real
                    if is_real_xplet([t1.hits[0].hit_id, t1.hits[1].hit_id, t1.hits[2].hit_id, t2.hits[2].hit_id]) == XpletType.REAL:
                        name = f"{t1.hits[0].hit_id}_{t1.hits[1].hit_id}_{t1.hits[2].hit_id}_{t2.hits[2].hit_id}"
                        hard_stats.append(f"qplet,{name},dcurv,{dcurv},")
                        if not cheat:
                            continue
                    else:
                        continue

                # ---- compute strength WITHOUT creating Quadruplet object
                # This replicates _compute_strength logic but uses only available info.
                # Note: must match your _compute_strength math exactly.
                xy_strength = 1.0 - ((dcurv / qplet_max_dcurv) ** self.config.xy_power)

                max_drz = t1.drz if t1.drz >= t2.drz else t2.drz
                rz_strength = 1.0 - ((max_drz / self.config.tplet_max_drz) ** self.config.rz_power)

                numerator = self.config.num_multiplier * (
                    self.config.xy_relative_strength * xy_strength +
                    (1.0 - self.config.xy_relative_strength) * rz_strength
                )

                # volayer_span for qplet: from first hit of t1 to last hit of t2
                volayer_span = t2.hits[-1].volayer - t1.hits[0].volayer
                exceeding_volayer_span = volayer_span - 4 + 1

                denominator = (1.0 + exceeding_volayer_span) ** self.config.volayer_power

                strength = numerator / denominator
                if self.config.strength_bounds is not None:
                    strength = np.clip(strength, *self.config.strength_bounds)

                # ---- EARLY CUT 2: strength cut
                if strength > qplet_max_strength:
                    if is_real_xplet([t1.hits[0].hit_id, t1.hits[1].hit_id, t1.hits[2].hit_id, t2.hits[2].hit_id]) == XpletType.REAL:
                        name = f"{t1.hits[0].hit_id}_{t1.hits[1].hit_id}_{t1.hits[2].hit_id}_{t2.hits[2].hit_id}"
                        hard_stats.append(f"qplet,{name},strength,{strength},")
                        if not cheat:
                            continue
                    else:
                        continue

                # If we are here: accepted -> create real Quadruplet object and wire it in
                qplet = Quadruplet(t1, t2)
                t1.outer.append(qplet)
                t2.inner.append(qplet)
                qplet.strength = strength
                quadruplets.append(qplet)

                if register_qubo:
                    self._register_qubo_quadruplet(qplet)

        self.logger.info(f"created {len(quadruplets)} quadruplets.")
        self.quadruplets = quadruplets


    def _is_invalid_triplet(self, tplet: Triplet) -> bool:
        # Apply hard cuts on triplets.
        # Currently, we look at three information:
        # * the number of layer miss
        # * the radius of the curvature formed by the three hits (cut on GeV)
        # * how well are the two doublets aligned in the R-Z plane

        is_real = self.dataw.is_real_xplet(tplet.hit_ids()) == XpletType.REAL

        # layer skips
        volayer_skip = tplet.hits[-1].volayer - tplet.hits[0].volayer
        if volayer_skip > self.config.max_layer_span + 1:
            if is_real:
                self.hard_cuts_stats.append(f'tplet,{tplet},volayer,{volayer_skip},')
                return not self.config.cheat
            return True
        # radius of curvature formed by the three hits
        if abs(tplet.curvature) > self.config.tplet_max_curv:
            if is_real:
                self.hard_cuts_stats.append(f'tplet,{tplet},curv,{tplet.curvature},')
                return not self.config.cheat
            return True
        # angle between the two doublets in the rz plane
        if tplet.drz > self.config.tplet_max_drz:
            if is_real:
                self.hard_cuts_stats.append(f'tplet,{tplet},drz,{tplet.drz},')
                return not self.config.cheat
            return True
        return False

    def _is_invalid_quadruplet(self, qplet: Quadruplet) -> bool:
        # Apply hard cuts on quadruplets.
        # Currently, we discard directly any potential quadruplet between triplets that don't have
        # a very similar curvature in the X-Y plane. Then, we compute the coupling strength (combining
        # layer miss, R-Z plane delta angles and curvature) and apply a cut on it.
        is_real = self.dataw.is_real_xplet(qplet.hit_ids()) == XpletType.REAL

        # delta delta curvature between the two triplets
        ret = qplet.delta_curvature > self.config.qplet_max_dcurv
        if ret and is_real:
            self.hard_cuts_stats.append(f'qplet,{qplet},dcurv,{qplet.delta_curvature},')
            return not self.config.cheat

        # strength of the quadruplet
        qplet.strength = self._compute_strength(qplet)
        ret = qplet.strength > self.config.qplet_max_strength
        if ret and is_real:
            self.hard_cuts_stats.append(f'qplet,{qplet},strength,{qplet.strength},')
            return not self.config.cheat
        return ret

    # --------------- qubo weights

    def _compute_weight(self, tplet: Triplet) -> float:
        # Just return a constant for now.
        # In the future, it would be interesting to try to measure a-priori how interesting a triplet is
        # (using for example the number of quadruplets it is part of) and encode this information into a
        # variable weight.
        return self.config.qubo_bias_weight

    def _compute_strength(self, qplet: Quadruplet) -> float:
        # Combine information about the layer miss, the alignment in the R-Z plane and the curvature in the X-Y plane.
        # The strength is negative, its range depending on the configuration (default: 1 >= strength >= max_strength)

        if qplet.strength != 0:  # avoid computing twice
            return qplet.strength

        # normalised difference of curvature between the two triplets
        xy_strength = 1 - ((qplet.delta_curvature / self.config.qplet_max_dcurv) ** self.config.xy_power)

        # normalised [maximum] angle in the R-Z plane
        max_drz = max(qplet.t1.drz, qplet.t2.drz)
        rz_strength = 1 - ((max_drz / self.config.tplet_max_drz) ** self.config.rz_power)

        # numerator: combine both X-Y and R-Z plane information
        numerator = self.config.num_multiplier * (
                self.config.xy_relative_strength * xy_strength +
                (1 - self.config.xy_relative_strength) * rz_strength
        )

        # denominator: shrink the strength proportional to the number of layer miss
        exceeding_volayer_span = qplet.volayer_span - len(qplet.hits) + 1
        denominator = (1 + exceeding_volayer_span) ** self.config.volayer_power

        strength = numerator / denominator

        # clip the strength if needed
        if self.config.strength_bounds is not None:
            strength = np.clip(strength, *self.config.strength_bounds)

        return strength

    def _compute_conflict_strength(self, t1: Triplet, t2: Triplet) -> float:
        # Just return a constant for now.
        # Careful: if too low, the number of remaining conflicts in the QUBO solution will explode.
        # If too high, qbsolv can behave strangely: the execution time augments significantly while the
        # scores drop slowly.
        return self.config.qubo_conflict_strength
