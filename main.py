import models
import torchinfo

model = models.get_model("retinafacenet_resnet50_fpn").eval()
# print(model.state_dict().keys())
#  bn_keys = []
#  for key in model.state_dict().keys():
    #  if "bn" in key:
        #  bn_keys.append(key)
#  
#  print(bn_keys)

torchinfo.summary(model,(1,3,224,224))
