import torch

selfsup_loss = torch.nn.BCELoss()
# remove predictions of 'dishes'
class_index = [0, 1, 2, 4, 5, 6, 7, 8, 9]

def sod_label_generation(self, labels):
    label_no_dishes = labels[:, class_index]
    labels_strong_sod = torch.sum(label_no_dishes, -2).clamp(min=0, max=2)

    return labels_strong_sod


def func_1(self, preds):
    preds_bool = preds > 0.5
    preds_bool_1 = preds_bool[:, class_index]
    index_new = torch.sum(preds_bool_1, axis=1).clamp(max=2)
    return index_new


def sod_2_label(self, sod_student):
    max_value, index = torch.max(sod_student, dim=-1)
    index_2 = torch.ge(index, 2)
    index_0 = torch.le(index, 0)
    index_1 = torch.eq(index, 1)
    index_new = torch.stack([index_0, index_1, index_2], dim=-1)
    return index_new

def cross_task_consistency(strong_preds_student, sod_preds):
    strong_preds_student_sum_clamp = sod_label_generation(strong_preds_student)
    sod_student_softmax = torch.softmax(sod_preds, dim=-1)
    sod_student_softmax_max, max_index = torch.max(sod_student_softmax, dim=-1)
    sod_student_tag = sod_student_softmax_max * max_index
    cross_task_consis_loss = selfsup_loss(
        strong_preds_student_sum_clamp, sod_student_tag
    )
    return cross_task_consis_loss


def con_loss(strong_preds_student, strong_preds_teacher,
             strong_mask, non_strong_mask,
             sod_mask_w_u_1, sod_mask_w_u_0):
    sed_student = strong_preds_student.permute(0, 2, 1)
    sed_teacher = strong_preds_teacher.permute(0, 2, 1)
    sed_student_w_u = sed_student[non_strong_mask]
    sed_teacher_w_u = sed_teacher[non_strong_mask]
    sed_student_w_u_1 = sod_mask_w_u_1 * sed_student_w_u
    sed_teacher_w_u_1 = sod_mask_w_u_1 * sed_teacher_w_u
    strong_self_sup_loss_new1 = selfsup_loss(
        sed_student_w_u_1, sed_teacher_w_u_1.detach()
    )
    sed_student_w_u_0 = sod_mask_w_u_0 * sed_student_w_u
    sed_teacher_w_u_0 = sod_mask_w_u_0 * sed_teacher_w_u

    strong_self_sup_loss_new0 = selfsup_loss(
        sed_student_w_u_0, sed_teacher_w_u_0.detach()
    )

    strong_self_sup_loss_strong = selfsup_loss(
        strong_preds_student[strong_mask], strong_preds_teacher[strong_mask].detach()
    )
    strong_self_sup_loss = strong_self_sup_loss_strong + strong_self_sup_loss_new0 * 0.3 + strong_self_sup_loss_new1

    return strong_self_sup_loss


def cross_task_post_processing(sed_preds, sod_preds):
    batch_size = sed_preds.shape[0]
    frame_size = sed_preds.shape[2]

    sed_preds_bool = sed_preds > 0.5

    class_index = [0, 1, 2, 4, 5, 6, 7, 8, 9]
    preds_bool_1 = sed_preds_bool[:, class_index]
    sed2vad_student_new = torch.sum(preds_bool_1, axis=1).clamp(min=0, max=2).long()

    max_value, sod_index = torch.max(sod_preds, dim=-1)  # vad_index: (batch, T)
    tmp_mask_1 = sod_index - sed2vad_student_new  # (batch, T)

    mask_vad_sed_great_1 = torch.eq(tmp_mask_1, 1)  # (batch, T)
    mask_vad_1 = torch.eq(sod_index, 1)  # (batch, T)
    mask_vad_2 = torch.eq(sod_index, 2)

    mask1 = mask_vad_1 * mask_vad_sed_great_1 + 0.  # (batch, T)
    mask2 = mask_vad_2 * mask_vad_sed_great_1 + 0.

    ######### the first situation ########
    strong_preds_1 = sed_preds[:, class_index]
    max_sed_value, _ = torch.max(strong_preds_1, dim=-2)
    max_sed_value = max_sed_value.unsqueeze(dim=1)  # (batch, 1, T)

    tmp2 = strong_preds_1 - max_sed_value

    tmp2_mask = torch.eq(tmp2, 0.) + 0.  # (batch, 9, T)

    mask1 = mask1.unsqueeze(dim=1)  # (batch, 1, T)

    tmp2_mask = tmp2_mask * mask1  # * mask1_num
    tmp2_mask = tmp2_mask * 0.2

    tmp2_mask_0_2 = tmp2_mask[:, :3, :]
    tmp2_mask_4_9 = tmp2_mask[:, 3:, :]

    tmp2_mask_3 = torch.zeros([batch_size, 1, frame_size], device='cuda')
    tmp2_mask = torch.cat([tmp2_mask_0_2, tmp2_mask_3, tmp2_mask_4_9], dim=1)

    ######## the second situation ########

    max_sed_value, indices = torch.topk(strong_preds_1, k=2, dim=-2)
    # max_sed_value: (batch, 2, T)
    second_max_sed_value = max_sed_value[:, 1, :]
    second_max_sed_greater_02 = (second_max_sed_value > 0.2) + 0.
    second_max_sed_value = second_max_sed_value * second_max_sed_greater_02
    second_max_sed_value = second_max_sed_value.unsqueeze(dim=1)  # (batch, 1, T)

    tmp2 = strong_preds_1 - second_max_sed_value
    tmp3_mask = torch.eq(tmp2, 0.) + 0.  # (batch, 9, T)
    mask2 = mask2.unsqueeze(dim=1)  # (batch, 1, T)
    tmp3_mask = tmp3_mask * mask2  # * mask2_num
    tmp3_mask = tmp3_mask * 0.2

    tmp3_mask_0_2 = tmp3_mask[:, :3, :]
    tmp3_mask_4_9 = tmp3_mask[:, 3:, :]
    tmp2_mask_3 = torch.zeros([batch_size, 1, frame_size], device='cuda')
    tmp3_mask = torch.cat([tmp3_mask_0_2, tmp2_mask_3, tmp3_mask_4_9], dim=1)

    sed_preds = (sed_preds + tmp2_mask + tmp3_mask).clamp(0., 1.)

    return sed_preds