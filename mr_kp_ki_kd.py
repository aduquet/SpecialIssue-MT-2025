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
        
    @staticmethod
    def mr_hlh_rt(rt_ref, rt):

        if rt_ref >= rt:
            return 1
        else:
            return 0
    
    @staticmethod    
    def mr_hlh_ess_step(ref_ess, ess_step):

        if ref_ess <= ess_step:
            return 1
        else:
            return 0
        
    @staticmethod
    def mr_hlh_eng(ref_eng_rel, eng_rel):
        if eng_rel > ref_eng_rel:
            return 1
        else:
            return 0
    
    # HLL 
    @staticmethod
    def mr_hll_rt(rt_ref, rt):

        if rt_ref >= rt:
            return 1
        else:
            return 0

    @staticmethod    
    def mr_hll_ess_step(ref_ess, ess_step):

        if ref_ess <= ess_step:
            return 1
        else:
            return 0

    @staticmethod    
    def mr_hll_ess_step(ref_ess, ess_step):

        if ref_ess <= ess_step:
            return 1
        else:
            return 0       