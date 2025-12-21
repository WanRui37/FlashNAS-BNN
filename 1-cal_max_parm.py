channel_ = [32] + [64] + [128] * 3 + [256] * 3 + [512] * 6 + [1024] * 2


def check_the_parm(channel_):
    parm_total = 0
    parm_single = 0
    for ii in range(len(channel_)):

        if ii == 0:
            parm_single = 3 * 3 * 3 * channel_[ii]
        else:
            parm_single = 3 * 3 * channel_[ii - 1] * channel_[ii]
            parm_single = 1 * 1 * channel_[ii - 1] * channel_[ii] * 2 + parm_single

        parm_total = parm_total + parm_single
    print(parm_total)


check_the_parm(channel_)
