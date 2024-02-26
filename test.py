from util import *
import random

test_code ='''
    hdr->hdr.it_version = PKTHDR_RADIOTAP_VERSION;
        hdr->hdr.it_pad = 0;
        hdr->hdr.it_len = cpu_to_le16(sizeof(hdr->hdr.it_present = __constant_cpu_to_le32(
             (1 << IEEE80211_RADIOTAP_FLAGS) |
             (1 << IEEE80211_RADIOTAP_RATE) |
             (1 << IEEE80211_RADIOTAP_CHANNEL));
        hdr->rt_flags = 0;
        hdr->rt_rate = txrate->bitrate / 5;
        hdr->rt_channel = data->channel->center_freq;
    '''
def random_test(fileDic, print_code = False):
    '''
        Random find a key from dictionary
        it will print the result if requires print the code

        args:
            fileDic: 
    '''
    code = random.choice(list(fileDic))
    selected_test(fileDic,code,print_code=print_code)
    return code

def selected_test(fileDic, code, print_code = True):
    if print_code:
        print(get_code_from_file(fileDic,code))
        print("-----------------")
        dic_single_print(fileDic,code)
    return code

