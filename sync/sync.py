def read_sync_file(file_path):
    pos_t = {}
    t_pos = {}
    read = False
    n = 0
    for line in open(file_path):
        if n == 10:
            break
        if line == "<BODY>\n":
            read = True
            continue
        if line == "</BODY>\n":
            break
        if read:
            n += 1
            pos_mat_time = line.split("Start=")[1].split(">")[0]
            t_file_time = line.split("ENUSCC>")[1].split("<")[0]
            pos_t[int(pos_mat_time)] = int(t_file_time)
            t_pos[int(t_file_time)] = int(pos_mat_time)
    return pos_t, t_pos