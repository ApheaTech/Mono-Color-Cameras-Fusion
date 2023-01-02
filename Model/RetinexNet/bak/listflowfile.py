import os


def dataloader(rootpath):
    # all_color = []
    # all_mono = []
    # subdir_list = ['a_rain_of_stones_x2', 'eating_camera2_x2', 'flower_storm_x2', 'funnyworld_augmented0_x2',
    #                'lonetree_augmented0_x2', 'lonetree_difftex_x2', 'lonetree_winter_x2']
    # for subdir in subdir_list:
    #     for i in range(1, 501):
    #         filepath = os.path.join(rootpath, subdir)
    #         if not os.path.exists(filepath + "/left/" + "%04d" % i + ".webp"):
    #             continue
    #
    #         all_color.append(filepath + "/left/" + "%04d" % i + ".webp")
    #         all_mono.append(filepath + "/right/" + "%04d" % i + ".webp")
    #
    # return all_color, all_mono
    all_low_img = []
    all_high_img = []
    filepath = os.path.join(rootpath, 'our485')
    low_path = os.path.join(filepath, 'low')
    high_path = os.path.join(filepath, 'high')
    for img_name in os.listdir(low_path):
        all_low_img.append(os.path.join(low_path, img_name))
        all_high_img.append(os.path.join(high_path, img_name))

    return all_low_img, all_high_img


# a_rain_of_stones_x2 eating_camera2_x2 flower_storm_x2 funnyworld_augmented0_x2 lonetree_augmented0_x2
# lonetree_difftex_x2 lonetree_winter_x2