# -*- coding: utf-8 -*-

if __name__ == '__main__':
    with open('demo.txt', 'r', encoding="utf-8") as file:
        demo = file.readlines()#按行读demo

    with open('beam.txt', 'r', encoding="utf-8") as file:
        beam = file.readlines()#按行读beam

    with open('README.template', 'r', encoding="utf-8") as file:
        template = file.readlines()#按行读template

    template = ''.join(template)#将template以空格连接

    for i in range(20):
        template = template.replace('[{}]'.format(i), demo[i].strip())#把template中的所有[]内的{}占位符用demo中去除首尾空格后的对应行替换

    for i in range(0, 10):
        beam_data = [line.strip() for line in beam[i * 4:(i + 1) * 4]]#beam中去掉首尾空格每四行作为一个列表存到beam_data
        beam_text = '<br>'.join(beam_data)#将beam_data以换行符连接存到bean_text
        template = template.replace('({})'.format(i), beam_text)#把template中的所有()内的{}占位符用beam_text替换

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)#写回template
