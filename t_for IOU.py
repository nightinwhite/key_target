from utils import *
a = [113.10862731933594, 181.96922302246094, 16.613776623817294, 32.78420913142364]
b = [159.1591339111328, 256.27764892578125, 265.0074325682069, -136.77631987319933]
print (rec_centre_To_rec_corner_L(a))
print (rec_centre_To_rec_corner_L(b))
print ()
print(calc_jaccard(rec_centre_To_rec_corner_L(a),rec_centre_To_rec_corner_L(b)))
