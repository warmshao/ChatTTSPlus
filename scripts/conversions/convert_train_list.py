# -*- coding: utf-8 -*-
# @Time    : 2024/12/18
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: convert_train_list.py
"""
python scripts/conversions/convert_train_list.py \
data/leijun/asr_opt/denoise_opt.list \
data/leijun/asr_opt/denoise_opt_new.list \
leijun
"""
import argparse
import pdb


def convert_list(input_file, output_file, speaker_name):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 去掉行末的换行符
            line = line.strip()
            if line:
                # 分割行内容
                parts = line.split('|')
                audio_path = parts[0]
                lang = parts[-2]
                text = parts[-1]
                # 构建新的行，没有空格
                new_line = f"{speaker_name}|{audio_path}|{lang}|{text}\n"
                outfile.write(new_line)
    print(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert list file format.')
    parser.add_argument('input_file', type=str, help='Input list file')
    parser.add_argument('output_file', type=str, help='Output list file')
    parser.add_argument('speaker_name', type=str, help='Speaker name to add to each line')

    args = parser.parse_args()

    convert_list(args.input_file, args.output_file, args.speaker_name)
