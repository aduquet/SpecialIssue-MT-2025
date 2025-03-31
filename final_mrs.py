import math

class mrs_pid:

    @staticmethod
    def mr_PR(PR):
        if 1.0 <= PR <= 1.05:
            return 1
        else:
            return 0
        
    @staticmethod
    def mr_rt_f(rt_ref, rt):

        if rt_ref >= rt:
            return 1
        else:
            return 0
    
    @staticmethod
    def mr_rt_s(rt_ref, rt):

        if rt_ref <= rt:
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
    def mr_osc(osc, oscs):
        if osc == 1 and oscs <5:
            return 1
        else:
            return 0
        
    @staticmethod    
    def mr_ess_step_g(ref_ess, ess_step):

        if ref_ess <= ess_step:
            return 1
        else:
            return 0
        
    @staticmethod    
    def mr_ess_step_l(ref_ess, ess_step):

        if ref_ess >= ess_step:
            return 1
        else:
            return 0
        
    @staticmethod
    def mr_IAE_l(ref_IAE, IAE):
        if ref_IAE >= IAE:
            return 1
        else:
            return 0

    @staticmethod
    def mr_IAE_g(ref_IAE, IAE):
        if ref_IAE <= IAE:
            return 1
        else:
            return 0
        
    @staticmethod
    def mr_ITAE_l(ref_ITAE, ITAE):
        if ref_ITAE >= ITAE:
            return 1
        else:
            return 0

    @staticmethod
    def mr_ITAE_g(ref_ITAE, ITAE):
        if ref_ITAE <= ITAE:
            return 1
        else:
            return 0

    @staticmethod
    def mr_eng_p_s_g(ref_eng_rel, eng_rel):
        if ref_eng_rel <= eng_rel:
            return 1
        else:
            return 0    
               
    @staticmethod
    def mr_eng_p_s_l(ref_eng_rel, eng_rel):
        if ref_eng_rel >= eng_rel:
            return 1
        else:
            return 0    