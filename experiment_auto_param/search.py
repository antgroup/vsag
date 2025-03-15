import os

dataset_s = [
    "gist-960-euclidean"
    # "sift-128-euclidean",
    # "fashion-mnist-784-euclidean",
    # "glove-100-angular"
]

M_s = [8, 12, 16, 20, 24, 28, 32]
A_s = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

config_s = []
for dataset in dataset_s:
    for M in M_s:
        for A in A_s:
            config_s.append([dataset, M, A])

core = 0
for config in config_s:
    for mode in ["auto"]:
        dataset = config[0]
        m_c = config[1]
        m_s = config[1] * 2
        a_c = config[2]
        a_s = config[2]
        is_auto = 0

        if mode == "auto":
            m_c = 32
            a_c = 2.0
            is_auto = 1

        command = (f"/tbase-project/vsag/build-release/experiment/benchmark_build_index "
                   f"{dataset} 4 -1 0 {is_auto} {m_c} {a_c} {m_s} {a_s} 300 > "
                   f"/tbase-project/vsag/experiment_auto_param/search/{mode}_314_2/search_{dataset}_C{m_c}_{a_c}_300_S{m_s}_{a_s}_-1_{mode}.txt ")
        print(command)
        os.system(command)
        core += 1
