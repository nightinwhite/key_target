#coding:utf-8
import xml.dom.minidom
# 100000001 目标,100000002 迷彩目标,100000003 一般目标,
# 100000004 迷彩车辆,100000005 迷彩建筑,100000006 一般车辆,100000007 一般建筑,100000008 自然背景
# 100000009 迷彩其它,100000010 非迷彩其它
# 迷彩车辆 0 ，迷彩建筑 1， 迷彩其它 2，
# 非迷彩车辆 3 ，非迷彩建筑 4， 非迷彩其它 5， 自然背景 6

ID_to_Class_Idx = {100000004: 0,
                   100000005: 1,
                   100000009: 2,
                   100000006: 3,
                   100000007: 4,
                   100000010: 5,
                   100000008: 6,
                   }

def parse_xml(xml_path):
    anns_list = []
    DOM_Tree = xml.dom.minidom.parse(xml_path)
    collcetion = DOM_Tree.documentElement
    anns = collcetion.getElementsByTagName("HRSC_Object")
    for ann in anns:
        tmp_data = []
        Class_ID = ann.getElementsByTagName("Class_ID")
        mbox_cx = ann.getElementsByTagName("mbox_cx")
        mbox_cy = ann.getElementsByTagName("mbox_cy")
        mbox_w = ann.getElementsByTagName("mbox_w")
        mbox_h = ann.getElementsByTagName("mbox_h")
        mbox_ang = ann.getElementsByTagName("mbox_ang")

        Class_ID_value = int(Class_ID[0].childNodes[0].data)
        try:
            mbox_cx_value = float(mbox_cx[0].childNodes[0].data)
            mbox_cy_value = float(mbox_cy[0].childNodes[0].data)
            mbox_w_value = float(mbox_w[0].childNodes[0].data)
            mbox_h_value = float(mbox_h[0].childNodes[0].data)
            mbox_ang_value = float(mbox_ang[0].childNodes[0].data)
        except ValueError as e:
            continue
        tmp_data = [ID_to_Class_Idx.get(Class_ID_value, -1), [mbox_cx_value, mbox_cy_value, mbox_w_value, mbox_h_value, mbox_ang_value]]
        anns_list.append(tmp_data)
    return anns_list

if __name__ == '__main__':
    tmp_path = "tst_data/mn_380.xml"
    anns = parse_xml(tmp_path)
    print(anns[0])