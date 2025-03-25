import math

class mrs:

    @staticmethod
    def mr_lll_one(rt_ref, rt):

        if rt_ref >= rt:
            return 1
        else:
            return 0
        
    @staticmethod
    def mr_lll_two(eng_ref, eng):
        if eng_ref > eng:
            return 1
        else:
            return 0


    @staticmethod
    def mr_lll_tree(osc, oscs):
        if osc == 1 and oscs <5:
            return 1
        else:
            return 0

    @staticmethod
    def mr_PR(PR):
        if 1.0 <= PR <= 1.05:
            return 1
        else:
            return 0
        
    @staticmethod
    def mr_AIE(ref_IAE, IAE):
        if ref_IAE >= IAE:
            return 1
        else:
            return 0
        
