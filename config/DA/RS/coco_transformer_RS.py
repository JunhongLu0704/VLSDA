data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None

# ----clip setting
backend = 'open_clip'  # ['open_clip', 'long_clip']
clip_model_name = 'ViT-B-32'
pretrained = 'GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt'  # your VLM path
clip_dim = 512
shift_range = 1.2
small_size = 48.0

da_backbone_loss_coef = 0.20
loss_VLPA_coef = 0.1
loss_GCCA_coef = 0.0025
temperature = 0.05


lr = 0.0001
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
param_dict_type = 'ddetr_in_mmdet'
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2
weight_decay = 0.0001
burn_epochs = 30  # 自训练起始epoch
epochs = 46
lr_drop = 30  # 第XX个,例如 30则为epoch=30时学习率下降，epoch从0开始(0-35)
save_checkpoint_interval = 100
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [33, 45]

modelname = 'dino'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = True

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

decoder_sa_type = 'sa'  # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher'  # or SimpleMinsumMatcher
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 4
dn_scalar = 100
dn_label_coef = 1.0
dn_bbox_coef = 1.0

match_unstable_error = True

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0
use_detached_boxes_dec_out = False

# for self-training
# self_training = True
strong_aug = True  # 目标域数据强增广
pseudo_label_threshold = 0.3  # 伪标签置信度阈值
ema_decay_teacher = 0.999
ema_decay_best_model = 0.9
self_training_loss_coef = 1.0
