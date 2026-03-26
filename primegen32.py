import numpy as np
import argparse


def log_diff(a, b):
    return abs(np.log2(a) - np.log2(b))


class PrimeList:

    def __init__(self, log_scale_min: float, log_scale_max: float,
                 list_low: list, list_high: list):
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.list_low = np.array(list_low)
        self.list_high = np.array(list_high)
        self.__next_pos = 1

    def interpret_pos(self) -> float:
        # Perform bit reversed traversal (sort of)
        # For example, if self.__next_pos = 10, we will select
        # len(list_low) * 0.0101 (binary fraction)
        # as the next position
        # we increment self.__next_pos by 1 for every sample_many_primes execution.
        # Next, self.__next_pos will be set to 11, and we will select
        # len(list_low) * 0.1101 (binary fraction)
        # as the next position
        pos = self.__next_pos
        pos_interpreted = 0.0
        incr = 0.5
        ratio = 0.5
        while pos > 0:
            if pos & 1:
                pos_interpreted += incr
            incr *= ratio
            pos >>= 1
        return pos_interpreted

    def sample_aux(self, num: int) -> list:
        len_list_high = len(self.list_high)
        aux_list = []
        for i in range(min(num, len_list_high)):
            aux_list.append(self.list_high[-1])
            self.list_high = np.delete(self.list_high, -1)

        if num > len_list_high:
            for i in range(num - len_list_high):
                aux_list.append(self.list_low[-1])
                self.list_low = np.delete(self.list_low, -1)
        return aux_list

    def sample_many_primes(self, target_log_scale: float, num: int) -> list:
        if num == 0:
            return []
        if num < 0:
            raise ValueError("num should be positive")
        primes = []

        if num == 1:
            primes.append(self.find_best_match(target_log_scale))
            return primes

        # Basically, we want to do better than greedy search.
        # Greedy search, where we sample primes one by one,
        # is likely to dissipate the primes close to a specific value fast,
        # resulting in more severe scale divergence.

        # We use the following heuristic:
        avg_log_scale = target_log_scale / num
        # if avg_log_scale is not in the prime range, just sample one by one
        if avg_log_scale < self.log_scale_min or avg_log_scale > self.log_scale_max:
            for i in range(num):
                primes.append(self.find_best_match(avg_log_scale))
            return primes

        # Otherwise, we select a subrange to sample primes, which is determined as
        # (avg_log_scale - range_width, avg_log_scale + range_width)
        # First, we make sure the subrange does not go beyond the prime range.
        range_width = min(abs(avg_log_scale - self.log_scale_min),
                          abs(avg_log_scale - self.log_scale_max))

        # Then, we use self.__next_pos
        # to create a somewhat random ratio between 0 and 1
        # and multiply the ratio to the range_width
        ratio = self.interpret_pos()
        # adjust the range such that it is in between 0.5 and 1.0
        # again, completely heuristic
        ratio = 0.5 + 0.5 * ratio

        range_width *= ratio
        self.__next_pos += 1

        # Then, sample primes placed at roughly equal intervals
        # within the subrange; i.e.,
        # p_0, p_1, ..., p_{num - 1} are the sampled primes
        # log(p_0) ~= avg_log_scale - range_width
        # log(p_{num - 1}) ~= avg_log_scale + range_width
        # and log(p_i) - log(p_{i - 1}) ~= (2 * range_width) / (num - 1)

        interval = 2 * range_width / (num - 1)
        target_prime_log_scale_list = [
            avg_log_scale - range_width + i * interval for i in range(num)
        ]

        for i in range(num):
            target_prime_scale = target_prime_log_scale_list[0]
            sampled = self.find_best_match(target_prime_scale)
            primes.append(sampled)
            if i == num - 1:
                break

            # Select a moving target because the scale of the primes will not
            # be the same as the target
            log_scale_diff = np.log2(sampled) - target_prime_scale
            log_scale_diff /= (num - i - 1)
            new_target_prime_log_scale_list = [
                target_prime_log_scale_list[i + 1] + log_scale_diff
                for i in range(num - i - 1)
            ]
            target_prime_log_scale_list = new_target_prime_log_scale_list

        return primes

    def find_best_match(self, log_scale: float, pop: bool = True) -> int:
        if log_scale < self.log_scale_min:
            print("log_scale too low, but continuing anyway...")
        if log_scale > self.log_scale_max:
            print("log_scale too high, but continuing anyway...")

        prime_list = np.concatenate((self.list_low, self.list_high))

        target = np.exp2(log_scale)
        low = 0
        high = len(prime_list) - 1
        best_match = prime_list[low]
        while low <= high:
            mid = (low + high) // 2
            if prime_list[mid] == target:
                best_match = prime_list[mid]
                break
            elif prime_list[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
            if log_diff(prime_list[mid], target) < log_diff(best_match, target):
                best_match = prime_list[mid]
        if pop:
            self.list_low = np.delete(self.list_low,
                                      np.where(self.list_low == best_match))
            self.list_high = np.delete(self.list_high,
                                       np.where(self.list_high == best_match))
        return best_match


# primes in range (2^19, 2^24.8)
# 786433, 1179649, 2752513, 5767169, 6684673, 6946817, 7340033,
# 8257537, 8519681, 8650753, 10223617, 11272193, 12451841, 13238273,
# 13631489, 14155777, 14942209, 16121857, 16384001, 16515073,
# 19529729, 20054017, 20316161, 21495809, 21626881, 22806529,
# 23068673, 24772609, 26214401, 27000833, 27918337, 28311553, 28704769

prime_dict = {

    # primes in range (2^24.8, 2^25.2)
    25:
        PrimeList(24.8, 25.2,
                  [29884417, 30539777, 31326209, 32899073, 33292289],
                  [35389441, 36175873, 37224449, 38535169]),
    # primes in range (2^25.8, 2^26.2)
    26:
        PrimeList(25.8, 26.2, [
            58851329, 59637761, 60424193, 61734913, 63700993, 64749569, 65929217
        ], [
            67239937, 67502081, 68681729, 69206017, 70254593, 71434241,
            72613889, 72744961, 74711041, 75104257, 76677121, 77070337
        ]),
    # primes in range (2^26.8, 2^27.2)
    27:
        PrimeList(26.8, 27.2, [
            117964801, 118751233, 120324097, 120586241, 123863041, 125042689,
            125698049, 126222337, 126615553, 127664129, 127795201, 130809857,
            132120577
        ], [
            135135233, 136314881, 138412033, 140771329, 141557761, 142344193,
            144310273, 145489921, 147849217, 149815297, 150863873, 151257089,
            151388161, 152174593
        ]),
    # primes in range (2^27.8, 2^28.2)
    28:
        PrimeList(27.8, 28.2, [
            234356737, 235798529, 236584961, 236716033, 239337473, 239861761,
            240648193, 241827841, 244842497, 244973569, 245235713, 245760001,
            246415361, 249561089, 253100033, 253493249, 254279681, 256376833,
            256770049, 257949697, 258605057, 260571137, 260702209, 261488641,
            261881857, 263323649, 263454721, 264634369, 265420801, 268042241
        ], [
            269221889, 270532609, 270794753, 272760833, 274726913, 276037633,
            276430849, 277086209, 279838721, 281935873, 283508737, 284950529,
            285474817, 288882689, 290455553, 291373057, 292159489, 295305217,
            297664513, 299499521, 302252033, 302776321, 304218113, 304742401,
            305135617, 306708481, 307888129, 308150273
        ]),
    # primes in range (2^28.8, 2^29.2)
    29:
        PrimeList(28.8, 29.2, [
            467795969, 468320257, 468713473, 469762049, 471072769, 473694209,
            474611713, 475267073, 478937089, 483131393, 483524609, 483655681,
            485621761, 486408193, 487063553, 487456769, 488243201, 489422849,
            491388929, 492175361, 493879297, 494534657, 495452161, 498466817,
            498597889, 500432897, 501350401, 504496129, 508690433, 510001153,
            511967233, 512360449, 517079041, 517472257, 518520833, 518914049,
            521011201, 524812289, 524943361, 525729793, 526123009, 528351233,
            529268737, 529924097, 531628033, 532283393, 533463041, 536215553,
            536608769
        ], [
            537133057, 539754497, 540540929, 540672001, 541327361, 543031297,
            543293441, 547749889, 548012033, 549978113, 550371329, 551288833,
            552861697, 553254913, 555220993, 557187073, 558235649, 561774593,
            564658177, 566886401, 568066049, 569638913, 570163201, 570949633,
            572915713, 575275009, 576454657, 576716801, 580780033, 581959681,
            582746113, 583794689, 584581121, 586285057, 590479361, 590872577,
            591265793, 595591169, 597688321, 598081537, 604635137, 605028353,
            605552641, 606339073, 607518721, 607911937, 608305153, 608698369,
            610926593, 611450881, 611844097, 612237313, 612892673, 613285889,
            615776257
        ]),
    # primes in range (2^29.8, 2^30.2)
    30:
        PrimeList(29.8, 30.2, [
            935329793, 938475521, 939655169, 940572673, 942800897, 943718401,
            946339841, 948699137, 949616641, 950009857, 950403073, 951582721,
            952238081, 954335233, 955383809, 957349889, 958136321, 958922753,
            962592769, 962854913, 964558849, 967180289, 969146369, 971243521,
            971898881, 972029953, 974258177, 975175681, 975831041, 976224257,
            976355329, 977534977, 979107841, 980156417, 983826433, 984481793,
            985006081, 985661441, 988938241, 989986817, 991297537, 992083969,
            993263617, 993918977, 994705409, 995622913, 996278273, 998244353,
            999161857, 999424001, 999948289, 1000210433, 1004535809, 1005060097,
            1006108673, 1007681537, 1010565121, 1012006913, 1012924417,
            1014366209, 1015283713, 1016463361, 1018429441, 1019478017,
            1019609089, 1021444097, 1023148033, 1026162689, 1026686977,
            1029308417, 1030619137, 1031667713, 1032192001, 1034813441,
            1036779521, 1037303809, 1038745601, 1040056321, 1040842753,
            1042415617, 1043464193, 1045430273, 1048707073, 1049100289,
            1051721729, 1052508161, 1053818881, 1054212097, 1055260673,
            1056178177, 1056440321, 1060765697, 1062469633, 1062862849,
            1064697857, 1065484289, 1068236801, 1070727169, 1071513601,
            1073479681
        ], [
            1073872897, 1074266113, 1081212929, 1083703297, 1085276161,
            1086455809, 1087635457, 1088684033, 1089208321, 1089601537,
            1091043329, 1091174401, 1092616193, 1093533697, 1093795841,
            1093926913, 1094582273, 1095761921, 1099431937, 1100873729,
            1102053377, 1103626241, 1107296257, 1108738049, 1110048769,
            1111883777, 1113980929, 1115815937, 1116209153, 1117126657,
            1118568449, 1121058817, 1122500609, 1123418113, 1125646337,
            1126957057, 1127219201, 1127743489, 1131151361, 1133510657,
            1135476737, 1136263169, 1137049601, 1138753537, 1139146753,
            1142685697, 1143472129, 1144258561, 1146093569, 1146880001,
            1147273217, 1150025729, 1150156801, 1150550017, 1151336449,
            1153302529, 1153564673, 1155137537, 1156055041, 1156841473,
            1157103617, 1157890049, 1158676481, 1159200769, 1159856129,
            1161822209, 1166934017, 1168900097, 1169817601, 1170604033,
            1172439041, 1172832257, 1173618689, 1174929409, 1175584769,
            1176764417, 1178861569, 1180434433, 1180827649, 1183449089,
            1184366593, 1184759809, 1185939457, 1188167681, 1188298753,
            1188560897, 1188954113, 1190264833, 1191837697, 1192230913,
            1193803777, 1196556289, 1199308801, 1200488449, 1201274881,
            1202454529, 1205862401, 1209139201, 1210318849, 1211105281,
            1212153857, 1212284929, 1214251009, 1215692801, 1218445313,
            1219362817, 1220411393, 1223950337, 1224081409, 1224736769,
            1226309633, 1227620353, 1228668929, 1231159297, 1231552513,
            1232601089
        ]),
    # primes in range (2^30.9, 2^31)
    # reserved for special primes (or sometimes, CtS)
    31:
        PrimeList(30.9, 31.0, [
            2009333761, 2010382337, 2010513409, 2011299841, 2013265921,
            2015887361, 2017067009, 2017984513, 2019950593, 2022309889,
            2022965249, 2023489537, 2025848833, 2027028481, 2028863489,
            2032402433, 2032795649, 2033975297, 2034892801, 2035286017,
            2035548161, 2035941377, 2039611393, 2043150337, 2047082497,
            2047868929, 2049048577, 2050490369, 2051407873, 2055602177,
            2055733249, 2055995393, 2061107201, 2062811137, 2067005441,
            2067398657, 2067529729, 2069495809, 2070151169, 2070675457,
            2070937601, 2075394049, 2077229057, 2080505857, 2081292289,
            2082078721, 2085093377, 2086404097, 2088763393, 2090336257,
            2090598401, 2091778049, 2094268417, 2094530561, 2094661633,
            2095054849, 2095710209, 2096889857, 2098593793, 2099249153,
            2100953089, 2102132737, 2102788097, 2106458113, 2107113473,
            2107506689, 2108817409, 2109603841, 2112225281, 2113011713,
            2113929217, 2114191361, 2114977793, 2117468161, 2119303169,
            2120482817, 2121793537, 2125725697, 2126118913, 2128740353,
            2130444289, 2130706433, 2132279297, 2134638593, 2135031809,
            2135162881, 2135818241, 2142502913, 2144468993, 2146041857,
            2146959361, 2147352577
        ], [])
}

# The below values are just for convenience
# and can be changed based on the word size requirements
main_log_scale = 30
sup_log_scale = 25
aux_scale = 31
supported_log_scale = [30, 35, 40, 45, 50]
supported_ds_scale = [52, 54, 56, 58, 60]


class State:

    def __init__(self,
                 ref_log_scale: int,
                 num_terminal: int = 0,
                 num_main: int = 0,
                 irregular_base: list = []):
        self.ref_log_scale = ref_log_scale
        self.num_terminal = num_terminal
        self.num_main = num_main
        self.irregular_base = irregular_base

    def copy(self):
        return State(self.ref_log_scale, self.num_terminal, self.num_main,
                     self.irregular_base.copy())

    def __str__(self):
        return f"{{{len(self.irregular_base) + self.num_main}, {self.num_terminal}}}"

    def get_next_state(self):
        # return the next state
        next_num_terminal = 0
        next_num_main = 0
        while True:
            # next_num_main * main_log_scale + next_num_terminal * sup_log_scale =
            # self.num_main * main_log_scale + self.num_terminal * sup_log_scale + ref_log_scale
            # (next_num_main - self.num_main) * main_log_scale =
            # ref_log_scale - (next_num_terminal - self.num_terminal) * sup_log_scale
            main_target_log_scale = self.ref_log_scale - (
                next_num_terminal - self.num_terminal) * sup_log_scale
            if main_target_log_scale % main_log_scale == 0:
                next_num_main = self.num_main + main_target_log_scale // main_log_scale
                break
            next_num_terminal += 1

        return State(self.ref_log_scale, next_num_terminal, next_num_main,
                     self.irregular_base.copy())


default_initial_state = {
    30: State(30, 0, 0, [786433, 1179649]),
    35: State(35, 2, 0, []),
    40: State(40, 2, 0, []),
    45: State(45, 0, 1, [8519681]),
    50: State(50, 0, 0, prime_dict[29].sample_many_primes(58, 2))
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BitPacker')
    parser.add_argument('--log_scale',
                        type=int,
                        default=35,
                        help='log scale for normal levels')
    parser.add_argument('--evalmod_log_scale',
                        type=int,
                        default=58,
                        help='log scale for EvalMod')
    parser.add_argument('--normal_level',
                        type=int,
                        default=19,
                        help='number of normal levels (+ StC)')
    parser.add_argument('--evalmod_level',
                        type=int,
                        default=8,
                        help='number of EvalMod levels')
    parser.add_argument('--cts_level',
                        type=int,
                        default=4,
                        help='number of CtS levels')
    parser.add_argument('--dnum',
                        type=int,
                        default=4,
                        help='decomposition number')
    parser.add_argument('--additional_terminal_base',
                        type=int,
                        default=0,
                        help='additional terminal base primes')
    parser.add_argument('--additional_main_base',
                        type=int,
                        default=0,
                        help='additional main base primes')

    args = parser.parse_args()
    log_scale = args.log_scale
    if log_scale not in supported_log_scale:
        raise ValueError(f"Log scale {log_scale} not supported")
    evalmod_log_scale = args.evalmod_log_scale
    if evalmod_log_scale not in supported_ds_scale:
        raise ValueError(f"EvalMod log scale {evalmod_log_scale} not supported")
    normal_level = args.normal_level
    evalmod_level = args.evalmod_level
    cts_level = args.cts_level
    dnum = args.dnum

    # maintained for (0 ~ normal_level)
    state_list = []
    log_scale_list = []

    terminal_primes = []
    irregular_base_primes = []
    main_primes = []

    # Initialize the configurations
    state_list.append(default_initial_state[log_scale].copy())
    state_list[0].num_terminal += args.additional_terminal_base
    state_list[0].num_main += args.additional_main_base

    log_scale_list.append(log_scale)

    terminal_primes += prime_dict[sup_log_scale].sample_many_primes(
        sup_log_scale * state_list[0].num_terminal, state_list[0].num_terminal)
    irregular_base_primes = state_list[0].irregular_base
    main_primes += prime_dict[main_log_scale].sample_many_primes(
        main_log_scale * state_list[0].num_main, state_list[0].num_main)

    max_num_main = state_list[0].num_main
    max_num_terminal = state_list[0].num_terminal

    for level in range(1, normal_level + 1):
        state_list.append(state_list[level - 1].get_next_state())

        # 1. if more supporting primes are needed, sample them
        if state_list[level].num_terminal > max_num_terminal:
            # sup primes are sampled one by one
            for i in range(state_list[level].num_terminal - max_num_terminal):
                terminal_primes.append(
                    prime_dict[sup_log_scale].find_best_match(sup_log_scale))
            max_num_terminal = state_list[level].num_terminal

        # 2. calculate the scale at next level. If more main primes are needed, sample them
        num_main_diff = state_list[level].num_main - state_list[level -
                                                                1].num_main
        num_terminal_diff = state_list[level].num_terminal - state_list[
            level - 1].num_terminal

        # scale_{level - 1} = scale_{level}^2 / prime_product (pp)
        # prime_product (pp) = product of primes added / product of primes discarded
        # we want to make scale_level as close to log_scale as possible
        log_pp_target = 2 * log_scale - log_scale_list[level - 1]

        log_pp = 0

        # 2-1. product of supporting primes
        if num_terminal_diff < 0:
            for i in range(state_list[level].num_terminal,
                           state_list[level - 1].num_terminal):
                log_pp -= np.log2(terminal_primes[i])
        elif num_terminal_diff > 0:
            for i in range(state_list[level - 1].num_terminal,
                           state_list[level].num_terminal):
                log_pp += np.log2(terminal_primes[i])

        # 2-2. product of main primes
        if num_main_diff < 0:
            for i in range(state_list[level].num_main,
                           state_list[level - 1].num_main):
                log_pp -= np.log2(main_primes[i])
        elif num_main_diff > 0:
            for i in range(state_list[level - 1].num_main,
                           min(state_list[level].num_main, max_num_main)):
                log_pp += np.log2(main_primes[i])
            # 2-3. sample main primes if needed
            if state_list[level].num_main > max_num_main:
                main_primes += prime_dict[main_log_scale].sample_many_primes(
                    log_pp_target - log_pp,
                    state_list[level].num_main - max_num_main)
                for i in range(max_num_main, state_list[level].num_main):
                    log_pp += np.log2(main_primes[i])
                max_num_main = state_list[level].num_main

        next_log_scale = 0.5 * (log_scale_list[level - 1] + log_pp)
        log_scale_list.append(next_log_scale)

        # check sanity
        if max_num_main != len(main_primes):
            raise ValueError("main primes length mismatch")
        if max_num_terminal != len(terminal_primes):
            raise ValueError("sup primes length mismatch")

    state_list_str = "{"
    for i in range(normal_level):
        state_list_str += str(state_list[i]) + ", "
    state_list_str += str(state_list[normal_level])

    evalmod_primes = []
    cts_primes = []

    if evalmod_level == 0 or cts_level == 0:
        state_list_str += "}"
        print("EvalMod or CtS levels are 0")
        print("I presume you are not supporting bootstrapping.")
        print()
        print("Normal log_scale list:", log_scale_list)
    elif max_num_main != state_list[normal_level].num_main:
        print("Current level cannot be the maximum normal level")
        print("Try adjusting --normal_level, --additional_terminal_base,",
              "or --additional_main_base")
        exit(1)
    else:
        evalmod_log_scale_list = [evalmod_log_scale]
        state_list_str += ", "

        # while we have used traversed the levels from the lowest to the highest,
        # we will traverse the levels from the highest to the lowest for EvalMod
        # Also, we simply use double-prime scaling for EvalMod

        evalmod_prime_pool = prime_dict[evalmod_log_scale // 2]
        for i in range(evalmod_level):
            target_log_pp = evalmod_log_scale_list[0] * 2 - evalmod_log_scale
            new_evalmod_primes = evalmod_prime_pool.sample_many_primes(
                target_log_pp, 2)
            new_log_scale = evalmod_log_scale_list[0] * 2 - np.log2(
                new_evalmod_primes[0]) - np.log2(new_evalmod_primes[1])
            evalmod_primes = new_evalmod_primes + evalmod_primes
            evalmod_log_scale_list = [new_log_scale] + evalmod_log_scale_list

        # Adding state list for EvalMod
        final_state = state_list[normal_level].copy()
        for i in range(evalmod_level):
            final_state.num_main += 2
            state_list_str += str(final_state) + ", "

        # CtS basically follows the evalmod scale, but we may need to lower it depending
        # on the number of supporting primes reduced in CtS
        num_terminal_primes_in_cts = max_num_terminal - state_list[
            normal_level].num_terminal
        num_cts_primes = 2 * cts_level - num_terminal_primes_in_cts
        cts_prime_log_scale = (
            evalmod_log_scale * cts_level -
            num_terminal_primes_in_cts * sup_log_scale) // num_cts_primes
        if cts_prime_log_scale >= aux_scale:
            cts_prime_log_scale = aux_scale - 1
        # Just perform greedy search for CtS primes
        for i in range(num_cts_primes):
            cts_primes.append(prime_dict[cts_prime_log_scale].find_best_match(
                cts_prime_log_scale))

        actual_cts_log_scale = 0
        for i in range(state_list[normal_level].num_terminal, max_num_terminal):
            actual_cts_log_scale += np.log2(terminal_primes[i])
        for i in range(num_cts_primes):
            actual_cts_log_scale += np.log2(cts_primes[i])
        actual_cts_log_scale /= (cts_level)

        # Adding state list for CtS
        num_cts_primes_left = num_cts_primes
        if num_cts_primes_left == 1:
            print("Failed! Try increasing cts levels")
            exit(1)
        compensation = 0
        for i in range(cts_level):
            # it's kind of tricky here...
            # For the current time being, we cannot perform rescaling with both reduced number of main and ter primes
            # So, we delay the rescaling for that prime
            if num_cts_primes_left == 3:
                final_state.num_main += 3
                num_cts_primes_left -= 3
                compensation = 1
            elif num_cts_primes_left >= 2:
                if compensation != 0:
                    print("Something went wrong...")
                final_state.num_main += 2
                num_cts_primes_left -= 2
            elif num_cts_primes_left == 1:
                print("Something went wrong...")
                exit(1)
            else:  # num_cts_primes_lest == 0
                if compensation != 0:
                    final_state.num_terminal += 1
                else:
                    final_state.num_terminal += 2
                compensation = 0
            state_list_str += str(final_state)
            if i != cts_level - 1:
                state_list_str += ", "
        state_list_str += "}"

        print("Normal log_scale list:", log_scale_list)
        print("EvalMod log_scale list:", evalmod_log_scale_list)
        print(f"CtS log_scale: {sup_log_scale} * {num_terminal_primes_in_cts}",
              f"+ {cts_prime_log_scale} * {num_cts_primes}",
              f"~= {actual_cts_log_scale} * {cts_level}")

    q_primes = irregular_base_primes + main_primes + evalmod_primes + cts_primes
    num_q_t = len(q_primes) + len(terminal_primes)
    num_aux = (num_q_t + dnum - 1) // dnum

    aux_primes = prime_dict[aux_scale].sample_aux(num_aux)

    print("Q primes:", q_primes)
    print("T primes:", terminal_primes)
    print("P primes:", aux_primes)
    print("State list (num_main + irr_base, num_terminal):", state_list_str)

    log_q = 0
    log_p = 0
    for q in terminal_primes:
        log_q += np.log2(q)
    for q in q_primes:
        log_q += np.log2(q)
    for p in aux_primes:
        log_p += np.log2(p)

    print("Num Q primes:", num_q_t)
    print("Num P primes:", num_aux)
    print("dnum:", dnum)
    print("logQ:", log_q)
    print("logP:", log_p)
    print("logPQ:", log_p + log_q)
