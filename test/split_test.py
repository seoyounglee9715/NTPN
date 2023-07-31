# from nuscenes.eval.prediction.splits import get_prediction_challenge_split
import sys
# from prediction.splits import get_prediction_challenge_split
sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\models')
from splits import get_prediction_challenge_split2

data_root = 'E:\\NuScenes\\trainval\\data\\sets\\nuscenes'
dataset = get_prediction_challenge_split2('mini_test', dataroot=data_root)


print(dataset) # 
print(len(dataset)) # mini_test: 71, val:9041
print(type(dataset))

