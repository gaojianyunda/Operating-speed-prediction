import torch

def compute_knowledge_enhanced_loss(predicted_features, R_H=1000.0,G_V=0.01, f_max=0.4, e_s=2.0, b=2.0, h_g=1.0,
                                     vS_prev_sum=1.0, vS_curr_sum=1.0, V_85=100.0, V_e=100.0,
                                     V_dc_a_vn=1.0, kk=0.01, device='cpu', aa=0.001, bb=0.01):
    
    # 计算 V_ds
    V_ds = calculate_V_ds(R_H, f_max, e_s, b, h_g, device)

    # 计算 V_dc
    V_dc = calculate_V_dc(V_dc_a_vn, kk, device)

    # 计算 a_de
    a_de = calculate_a_de(vS_prev_sum, vS_curr_sum, V_85, V_e, device)

    # 计算 a_vd
    a_vd = calculate_a_vd(V_dc, 0.01, 2000, G_V,device=device)  # κ 暂时设为 0.01

    ds = torch.tensor(100, device=device)

    # 假设 predicted_features 的形状为 [batch_size, num_features]
    batch_size, num_features = predicted_features.shape
    y_q_pred_previous = torch.zeros(batch_size, num_features, device=device)
    y_q_pred_previous[:, 1:] = predicted_features[:, :-1]

    L_knwl = (aa * torch.sum((predicted_features - torch.min(V_ds, V_dc)) ** 2) +
               bb * torch.sum(((predicted_features ** 2 - y_q_pred_previous ** 2) / (2 * ds) - torch.min(a_de, a_vd)) ** 2))
    
    return L_knwl

def calculate_V_ds(R_H=1000.0, f_max=0.4, e_s=2.0, b=2.0, h_g=1.0, device='cpu'):
    # 将输入转换为 Tensor
    R_H = torch.tensor(R_H, device=device)
    f_max = torch.tensor(f_max, device=device)
    e_s = torch.tensor(e_s, device=device)
    b = torch.tensor(b, device=device)
    h_g = torch.tensor(h_g, device=device)

    non_skidding = torch.sqrt(R_H / (127 * (f_max + 0.01 * e_s)))
    non_rollover = torch.sqrt(R_H / (127 * (b / (2 * h_g) + 0.01 * e_s)))
    V_ds = torch.min(non_skidding, non_rollover).to(device)
    return V_ds


def calculate_a_de(vS_prev_sum, vS_curr_sum, V_85, V_e, device='cpu'):
    ratio = vS_prev_sum / vS_curr_sum
    
    if ratio <= 0.5:
        a_de = torch.tensor(0.5, device=device)  # a_de > 0
    elif 0.5 < ratio < 2:
        a_de = torch.tensor(V_85, device=device)  # 确保 V_85 是 Tensor
    else:
        a_de = torch.tensor(-0.5, device=device)  # a_de < 0

    vehicle_type = 'car' 
    if vehicle_type == 'car':
        a_de = torch.clamp(a_de, min=-0.5, max=0.5)
    elif vehicle_type == 'truck':
        a_de = torch.clamp(a_de, min=-0.25, max=0.25)

    return a_de


def calculate_a_vd(V_dc, κ, P_e, η=0.90, μ=2000, V_85_0=20, C_a=0.3, C_rr=0.015, C_s=0.1, G_V=0, device='cpu'):
    a_vn = (V_dc ** 2) * κ  # 计算正常加速度
    air_density = 1.207  # kg/m^3 at 20 °C
    A_v = 2.0  # 车辆的横截面积，假设为小型车
    a_vt = (P_e * η) / (μ * V_85_0) - ((air_density * A_v * C_a * V_85_0 ** 2) / (2 * μ ** 2) + 
                                         (C_rr + μ * C_s) * 9.81 + G_V)

    a_vd = torch.sqrt(a_vn ** 2 + a_vt ** 2)
    return a_vd.to(device)

def calculate_V_dc(a_vn=1.0, kk=0.01, device='cpu'):
    # 将输入转换为 Tensor
    a_vn = torch.tensor(a_vn, device=device)
    kk = torch.tensor(kk, device=device)
    
    # 根据公式 V_dc = √(a_vn / κ)
    V_dc = torch.sqrt(a_vn / kk)

    return V_dc.to(device)

