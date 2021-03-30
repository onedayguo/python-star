import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

os.chdir('D:/GitHub/python-star/study/file')
path = 'D:/GitHub/python-star/study/file'


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('album'):
            value = (
                     root.find('category_id').text,
                     root.find('cate_code').text,
                     root.find('updated').text,
                     member[0].text,
                     member[1].text,
                     member[2].text,
                     member[3].text,
                     member[4].text,
                     member[5].text,
                     member[6].text,
                     member[7].text,
                     member[8].text,
                     member[9].text,
                     member[10].text,
                     member[11].text,
                     member[12].text,
                     member[13].text,
                     member[14].text,
                     member[15].text,
                     member[16].text,
                     member[17].text,
                     member[18].text,
                     member[19].text,
                     member[20].text,
                     member[21].text,
                     member[22].text,
                     member[23].text,
                     member[24].text,
                     member[25].text,
                     member[26].text,
                     member[27].text,
                     member[28].text,
                     member[29].text,
                     member[30].text,
                     member[31][0][0].text,
                     member[31][0][1].text,
                     member[31][0][2].text,
                     member[31][0][3].text,
                     member[31][0][4].text,
                     member[31][0][5].text,
                     member[31][0][6].text,
                     member[31][0][7].text,
                     member[31][0][8].text,
                     member[31][0][9].text,
                     member[31][0][10].text,
                     member[31][0][11].text,
                     member[31][0][12].text,
                     member[31][0][13].text,
                     member[31][0][14].text,
                     member[31][0][15][0].text,
                     member[31][0][15][1].text,
                     member[31][0][15][2].text
                     )
            xml_list.append(value)
            print(xml_list)
    column_name = [
        'category_id',
        'cate_code',
        'updated',
        'album_id',
        'album_name',
        'album_verpic',
        'album_verpic_extend',
        'album_horpic',
        'album_ver_small_pic',
        'album_hor_small_pic',
        'album_desc',
        'episode_updated',
        'episode_total',
        'genre',
        'area',
        'tvComment',
        'year',
        'language',
        'director',
        'actor',
        'publish_time',
        'fee',
        'is_clip',
        'update_time',
        'is_show',
        'is_early',
        'data_rights',
        'update_notification',
        'show_date',
        'score',
        'copyright_start_time',
        'copyright_end_time',
        'is_iptv',
        'is_dvb',
        'video_id',
        'video_name',
        'video_desc',
        'video_verpic',
        'video_horpic',
        'play_order',
        'update_time',
        'issue_time',
        'time_length',
        'video_url',
        'definition',
        'video_pic_16',
        'is_show',
        'fee',
        'tvPlayType',
        'dimension',
        'logo',
        'logoleft'
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = path
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('train.csv', index=None, encoding='utf_8_sig')
    print('Successfully converted xml to csv.')


main()