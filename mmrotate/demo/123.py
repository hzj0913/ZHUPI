import os
import glob
from shutil import copyfile

if __name__ == '__main__':
    file_path = '/data1/hzj/mmrotate/data/img_cut/images'
    json_list = glob.glob(file_path + '/*.json')
    save_path = 'E:/zjhu/img_cut/json'
    for j in json_list:
        # jname = os.path.basename(j)
        # print('j: ', j)
        # print('jname: ', jname)
        # copyfile(j, os.path.join(save_path, jname))
        os.remove(j)