import math

import numpy as np
import wx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wx.lib.masked import NumCtrl


class MyPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        # Створення полів введення
        self.a_entry = wx.TextCtrl(self)
        self.b_entry = wx.TextCtrl(self)
        self.c_entry = wx.TextCtrl(self)

        self.n_A = wx.TextCtrl(self)
        self.n_B = wx.TextCtrl(self)
        self.n_C = wx.TextCtrl(self)

        self.E_entry = wx.TextCtrl(self)
        self.nu_entry = wx.TextCtrl(self)
        self.P_entry = wx.TextCtrl(self)

        # Створення кнопки "Додати"
        self.all_points_button = wx.Button(self, label="Загальний графік")
        self.all_points_button.Bind(wx.EVT_BUTTON, self.on_all_points_button)

        # Розташування елементів на панелі
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, label="n_A:"), 0, wx.ALL, 5)
        sizer.Add(self.n_A, 0, wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="n_B:"), 0, wx.ALL, 5)
        sizer.Add(self.n_B, 0, wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="n_C:"), 0, wx.ALL, 5)
        sizer.Add(self.n_C, 0, wx.ALL, 5)

        sizer.Add(wx.StaticText(self, label="A:"), 0, wx.ALL, 5)
        sizer.Add(self.a_entry, 0, wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="B:"), 0, wx.ALL, 5)
        sizer.Add(self.b_entry, 0, wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="C:"), 0, wx.ALL, 5)
        sizer.Add(self.c_entry, 0, wx.ALL, 5)

        sizer.Add(wx.StaticText(self, label="E:"), 0, wx.ALL, 5)
        sizer.Add(self.E_entry, 0, wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="nu:"), 0, wx.ALL, 5)
        sizer.Add(self.nu_entry, 0, wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="P:"), 0, wx.ALL, 5)
        sizer.Add(self.P_entry, 0, wx.ALL, 5)

        sizer.Add(self.all_points_button, 0, wx.ALL, 5)

        self.SetSizer(sizer)

        self.SetSizer(sizer)

    def create_points(self, a, b, c, na, nb, nc):

        result = []

        step_a = a / na
        step_b = b / nb
        step_c = c / nc
        for k in range(nc):
            for j in range(nb):
                for i in range(na):
                    cube = self.create_cube(i * step_a, (i + 1) * step_a,
                                            j * step_b, (j + 1) * step_b,
                                            k * step_c, (k + 1) * step_c)
                    result.append(cube)

        return result

    def create_cube(self, a_start, a_end, b_start, b_end, c_start, c_end):
        a_size = a_end - a_start
        b_size = b_end - b_start
        c_size = c_end - c_start

        # x = [a_start, a_start + a_size / 2, a_end, a_start, a_end, a_start, a_start + a_size / 2, a_end, a_start,
        # a_end,
        #     a_start, a_end, a_start, a_start + a_size / 2, a_end, a_start, a_end, a_start, a_start + a_size / 2,
        #     a_end]
        # y = [b_start, b_start, b_start, b_start + b_size / 2, b_start + b_size / 2, b_end, b_end, b_end, b_start,
        #     b_start, b_end, b_end, b_start, b_start, b_start, b_start + b_size / 2, b_start + b_size / 2, b_end,
        #     b_end,
        #     b_end]
        # z = [c_start, c_start, c_start, c_start, c_start, c_start, c_start, c_start, c_start + c_size / 2,
        #    c_start + c_size / 2, c_start + c_size / 2, c_start + c_size / 2, c_end, c_end, c_end, c_end, c_end, c_end,
        #     c_end, c_end]

        x = [a_start,  # 1
             a_end,  # 2
             a_end,  # 3
             a_start,  # 4
             a_start,  # 5
             a_end,  # 6
             a_end,  # 7
             a_start,  # 8
             a_start + a_size / 2,  # 9
             a_end,  # 10
             a_start + a_size / 2,  # 11
             a_start,  # 12
             a_start,  # 13
             a_end,  # 14
             a_end,  # 15
             a_start,  # 16
             a_start + a_size / 2,  # 17
             a_end,  # 18
             a_start + a_size / 2,  # 19
             a_start  # 20
             ]
        y = [b_start,  # 1
             b_start,  # 2
             b_end,  # 3
             b_end,  # 4
             b_start,  # 5
             b_start,  # 6
             b_end,  # 7
             b_end,  # 8
             b_start,  # 9
             b_start + b_size / 2,  # 10
             b_end,  # 11
             b_start + b_size / 2,  # 12
             b_start,  # 13
             b_start,  # 14
             b_end,  # 15
             b_end,  # 16
             b_start,  # 17
             b_start + b_size / 2,  # 18
             b_end,  # 19
             b_start + b_size / 2  # 20
             ]
        z = [c_start,  # 1
             c_start,  # 2
             c_start,  # 3
             c_start,  # 4
             c_end,  # 5
             c_end,  # 6
             c_end,  # 7
             c_end,  # 8
             c_start,  # 9
             c_start,  # 10
             c_start,  # 11
             c_start,  # 12
             c_start + c_size / 2,  # 13
             c_start + c_size / 2,  # 14
             c_start + c_size / 2,  # 15
             c_start + c_size / 2,  # 16
             c_end,  # 17
             c_end,  # 18
             c_end,  # 19
             c_end  # 20
             ]
        result = []
        for i in range(20):
            result.append([x[i], y[i], z[i]])
        return result

    def separate_point(self, a, b, c, na, nb, nc):
        result = []
        step_a = a / na
        step_b = b / nb
        step_c = c / nc

        for k in range(2 * nc + 1):
            if k % 2 == 0:
                for j in range(2 * nb + 1):
                    if j % 2 == 0:
                        for i in range(2 * na + 1):
                            result.append([i * step_a / 2, j * step_b / 2, k * step_c / 2])
                    else:
                        for i in range(na + 1):
                            result.append([i * step_a, j * step_b / 2, k * step_c / 2])
            else:
                for j in range(nb + 1):
                    for i in range(na + 1):
                        result.append([i * step_a, j * step_b, k * step_c / 2])
        return result

    def NT_transform(self, akt, elem):
        result = []
        for cube in elem:
            nt_cube = []
            for i in cube:
                nt_cube.append(akt.index(i))
            result.append(nt_cube)
        return result

    def ZU_Chose(self, eleme):
        minim = min([sublist[2] for sublist in eleme])
        return [sublist for sublist in eleme if sublist[2] == minim]

    def ZP_Chose(self, eleme, side, side_of_axis):
        result = []
        if side == 1 or side == 3 or side == 5:
            minim = min([sublist[side_of_axis] for sublist in eleme])
            result = [sublist for sublist in eleme if sublist[side_of_axis] == minim]
        elif side == 2 or side == 4 or side == 6:
            maxim = max([sublist[side_of_axis] for sublist in eleme])
            result = [sublist for sublist in eleme if sublist[side_of_axis] == maxim]
        return result

    def DFIABG_Create(self):
        result = []
        for gamma in gamma_for:
            for beta in beta_for:
                for alpha in alpha_for:
                    a = []
                    for point in local_points:
                        if local_points.index(point) > 7:
                            a.append(self.DFIABD_center_side(alpha, beta, gamma, point[0], point[1], point[2]))
                        else:
                            a.append(self.DFIABD_angle(alpha, beta, gamma, point[0], point[1], point[2]))
                    result.append(a)
        return result

    def DFIABD_angle(self, alpha, beta, gamma, alpha_i, beta_i, gamma_i):
        result = [
            (1 / 8) * (1 + beta * beta_i) * (1 + gamma * gamma_i) *
            (alpha_i * (-2 + alpha * alpha_i + gamma * gamma_i + beta * beta_i) + alpha_i * (1 + alpha * alpha_i)),

            (1 / 8) * (1 + alpha * alpha_i) * (1 + gamma * gamma_i) *
            (beta_i * (-2 + alpha * alpha_i + gamma * gamma_i + beta * beta_i) + beta_i * (1 + beta * beta_i)),

            (1 / 8) * (1 + beta * beta_i) * (1 + alpha * alpha_i) *
            (gamma_i * (-2 + alpha * alpha_i + gamma * gamma_i + beta * beta_i) + gamma_i * (1 + gamma * gamma_i))
        ]
        return result

    def DFIABD_center_side(self, alpha, beta, gamma, alpha_i, beta_i, gamma_i):
        result = [
            (1 / 4) * (1 + beta * beta_i) * (1 + gamma * gamma_i) *
            (alpha_i * (
                    -beta_i * beta_i * gamma_i * gamma_i * alpha * alpha
                    - beta * beta * gamma_i * gamma_i * alpha_i * alpha_i
                    - beta_i * beta_i * gamma * gamma * alpha_i * alpha_i + 1) -
             (2 * beta_i * beta_i * gamma_i * gamma_i * alpha) * (alpha * alpha_i + 1)),

            (1 / 4) * (1 + alpha * alpha_i) * (1 + gamma * gamma_i) *
            (beta_i * (
                    -beta_i * beta_i * gamma_i * gamma_i * alpha * alpha
                    - beta * beta * gamma_i * gamma_i * alpha_i * alpha_i
                    - beta_i * beta_i * gamma * gamma * alpha_i * alpha_i + 1) -
             (2 * beta * gamma_i * gamma_i * alpha_i * alpha_i) * (beta_i * beta + 1)),

            (1 / 4) * (1 + beta * beta_i) * (1 + alpha * alpha_i) *
            (gamma_i * (
                    -beta_i * beta_i * gamma_i * gamma_i * alpha * alpha
                    - beta * beta * gamma_i * gamma_i * alpha_i * alpha_i
                    - beta_i * beta_i * gamma * gamma * alpha_i * alpha_i + 1) -
             (2 * beta_i * beta_i * gamma * alpha_i * alpha_i) * (gamma * gamma_i + 1))
        ]

        return result

    def DJ_Create(self, xyz):
        result = self.DExyzDEabg(xyz)
        return result

    def DELTA(self, xyz, alpha, beta, gamma):
        result = [
            self.DxyzDabg(xyz, alpha, beta, gamma, 0),
            self.DxyzDabg(xyz, alpha, beta, gamma, 1),
            self.DxyzDabg(xyz, alpha, beta, gamma, 2)
        ]
        return result

    def DExyzDEabg(self, xyz):
        result = []
        dfiabj = self.DFIABG_Create()
        for i in range(27):
            summ_x_a = []
            summ_x_b = []
            summ_x_g = []
            summ_y_a = []
            summ_y_b = []
            summ_y_g = []
            summ_z_a = []
            summ_z_b = []
            summ_z_g = []
            for point in xyz:
                index_of_nt = xyz.index(point)
                summ_x_a.append(point[0] * dfiabj[i][index_of_nt][0])
                summ_x_b.append(point[0] * dfiabj[i][index_of_nt][1])
                summ_x_g.append(point[0] * dfiabj[i][index_of_nt][2])

                summ_y_a.append(point[1] * dfiabj[i][index_of_nt][0])
                summ_y_b.append(point[1] * dfiabj[i][index_of_nt][1])
                summ_y_g.append(point[1] * dfiabj[i][index_of_nt][2])

                summ_z_a.append(point[2] * dfiabj[i][index_of_nt][0])
                summ_z_b.append(point[2] * dfiabj[i][index_of_nt][1])
                summ_z_g.append(point[2] * dfiabj[i][index_of_nt][2])
            result.append([
                [sum(summ_x_a), sum(summ_y_a), sum(summ_z_a)],
                [sum(summ_x_b), sum(summ_y_b), sum(summ_z_b)],
                [sum(summ_x_g), sum(summ_y_g), sum(summ_z_g)],
            ])

        return result

    def DxyzDabg(self, xyz, alpha, beta, gamma, index_of_abg):
        summ_x = 0
        summ_y = 0
        summ_z = 0
        for i in range(len(xyz)):
            if i > 7:
                centr_item = self.DFIABD_center_side(alpha, beta, gamma, local_points[i][0], local_points[i][1],
                                                     local_points[i][2])
                summ_x += xyz[i][0] * centr_item[index_of_abg]
                summ_y += xyz[i][1] * centr_item[index_of_abg]
                summ_z += xyz[i][2] * centr_item[index_of_abg]
            else:
                side_Item = self.DFIABD_angle(alpha, beta, gamma, local_points[i][0], local_points[i][1],
                                              local_points[i][2])
                summ_x += xyz[i][0] * side_Item[index_of_abg]
                summ_y += xyz[i][1] * side_Item[index_of_abg]
                summ_z += xyz[i][2] * side_Item[index_of_abg]

        return [summ_x, summ_y, summ_z]

    def calculate_determinant(self, a):
        det = a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1] - a[0][2] * a[1][
            1] * a[2][0] - a[0][0] * a[1][2] * a[2][1] - a[0][1] * a[1][0] * a[2][2]
        return det

    def solve_linear_equation(self, A, b):
        x = np.linalg.solve(A, b).tolist()
        return x

    def Solv_SLAR_for_elements(self, elements_cast, DJ_cast, DFIABG_cast):
        result = []
        for i in range(len(elements_cast)):
            result.append(self.Solv_SLAR_for_element(DJ_cast[i], DFIABG_cast))
        return result

    def Solv_SLAR_for_element(self, DJacobian, DFIABG_cast):
        result = []
        for delta_item in range(len(DJacobian)):
            dfixyz = []
            for points in DFIABG_cast[delta_item]:
                solved_L_E = self.solve_linear_equation(DJacobian[delta_item], points)
                dfixyz.append(solved_L_E)
            result.append(dfixyz)
        return result

    def calc_MGE(self, DFIXYZ_cast, determinant_list, c_list, lambda_val, nu_val, mu_val):
        matrix_a11 = []
        matrix_a22 = []
        matrix_a33 = []
        matrix_a12 = []
        matrix_a13 = []
        matrix_a23 = []
        for i in range(20):
            line_of_matrix_a11 = []
            line_of_matrix_a22 = []
            line_of_matrix_a33 = []
            line_of_matrix_a12 = []
            line_of_matrix_a13 = []
            line_of_matrix_a23 = []
            for j in range(20):
                a11 = []
                a22 = []
                a33 = []
                a12 = []
                a13 = []
                a23 = []
                general_index = 0
                for m in c_list:
                    for n in c_list:
                        for k in c_list:
                            dfi = DFIXYZ_cast[general_index]
                            a11.append(m * n * k *
                                       (lambda_val * (1 - nu_val) * (dfi[i][0] * dfi[j][0]) +
                                        mu_val * ((dfi[i][1] * dfi[j][1]) + (dfi[i][2] * dfi[j][2]))) *
                                       determinant_list[general_index])

                            a22.append(m * n * k *
                                       (lambda_val * (1 - nu_val) * (dfi[i][1] * dfi[j][1]) +
                                        mu_val * ((dfi[i][0] * dfi[j][0]) + (dfi[i][2] * dfi[j][2]))) *
                                       determinant_list[general_index])

                            a33.append(m * n * k *
                                       (lambda_val * (1 - nu_val) * (dfi[i][2] * dfi[j][2]) +
                                        mu_val * ((dfi[i][0] * dfi[j][0]) + (dfi[i][1] * dfi[j][1]))) *
                                       determinant_list[general_index])

                            a12.append(m * n * k * (lambda_val * nu_val * (dfi[i][0] * dfi[j][1]) +
                                                    mu_val * (dfi[i][1] * dfi[j][0])) * determinant_list[general_index])

                            a13.append(m * n * k * (lambda_val * nu_val * (dfi[i][0] * dfi[j][2]) +
                                                    mu_val * (dfi[i][2] * dfi[j][0])) * determinant_list[general_index])

                            a23.append(m * n * k * (lambda_val * nu_val * (dfi[i][1] * dfi[j][2]) +
                                                    mu_val * (dfi[i][2] * dfi[j][1])) * determinant_list[general_index])
                            general_index += 1
                line_of_matrix_a11.append(sum(a11))
                line_of_matrix_a22.append(sum(a22))
                line_of_matrix_a33.append(sum(a33))
                line_of_matrix_a12.append(sum(a12))
                line_of_matrix_a13.append(sum(a13))
                line_of_matrix_a23.append(sum(a23))
            matrix_a22.append(line_of_matrix_a22)
            matrix_a33.append(line_of_matrix_a33)
            matrix_a11.append(line_of_matrix_a11)
            matrix_a12.append(line_of_matrix_a12)
            matrix_a13.append(line_of_matrix_a13)
            matrix_a23.append(line_of_matrix_a23)

        matrix1 = np.array(matrix_a11)
        matrix2 = np.array(matrix_a12)
        matrix3 = np.array(matrix_a13)
        matrix4 = np.array(matrix_a22)
        matrix5 = np.array(matrix_a23)
        matrix6 = np.array(matrix_a33)

        big_matrix = np.zeros((60, 60))

        big_matrix[:20, :20] = matrix1
        big_matrix[:20, 20:40] = matrix2
        big_matrix[:20, 40:] = matrix3
        big_matrix[20:40, :20] = matrix2.T
        big_matrix[20:40, 20:40] = matrix4
        big_matrix[20:40, 40:] = matrix5
        big_matrix[40:, :20] = matrix3.T
        big_matrix[40:, 20:40] = matrix5.T
        big_matrix[40:, 40:] = matrix6

        big_matrix = big_matrix.tolist()
        return big_matrix

    def PSINT_angel(self, eta, tau, eta_i, tau_i):
        result = [
            (1 / 4) * (tau * tau_i + 1) * (eta_i * (eta_i * eta + tau_i * tau - 1) + eta_i * (eta_i * eta + 1)),
            (1 / 4) * (eta_i * eta + 1) * (tau_i * (eta_i * eta + tau_i * tau - 1) + tau_i * (tau_i * tau + 1))
        ]
        return result

    def PSINT_57(self, eta, tau, eta_i, tau_i):
        result = [
            (-tau * tau_i - 1) * eta,
            (1 / 2) * (1 - eta * eta) * tau_i
        ]
        return result

    def PSINT_68(self, eta, tau, eta_i, tau_i):
        result = [
            (1 / 2) * (1 - tau * tau) * eta_i,
            (-eta * eta_i - 1) * tau
        ]
        return result

    def PSINT_angel_main(self, eta, tau, eta_i, tau_i):
        result = (1 / 4) * (tau * tau_i + 1) * (eta * eta_i + 1) * (eta * eta_i + tau_i * tau - 1)
        return result

    def PSINT_57_main(self, eta, tau, eta_i, tau_i):
        result = (1 / 2) * (-eta * eta + 1) * (tau_i * tau + 1)
        return result

    def PSINT_68_main(self, eta, tau, eta_i, tau_i):
        result = (1 / 2) * (-tau * tau + 1) * (eta_i * eta + 1)
        return result

    def DEPSITE(self):
        result = []
        for eta in eta_for:
            for tau in tau_for:
                a = []
                for point in local_2d_points:
                    if local_2d_points.index(point) < 4:
                        a.append(self.PSINT_angel(eta, tau, point[0], point[1]))
                    elif local_2d_points.index(point) == 4 or local_2d_points.index(point) == 6:
                        a.append(self.PSINT_57(eta, tau, point[0], point[1]))
                    elif local_2d_points.index(point) == 5 or local_2d_points.index(point) == 7:
                        a.append(self.PSINT_68(eta, tau, point[0], point[1]))
                result.append(a)
        return result

    def DxyzDnt(self, xyz):
        result = []
        depsite = self.DEPSITE()
        index_for_depsite = 0
        for eta in eta_for:
            for tau in tau_for:
                summ_x_eta = []
                summ_y_eta = []
                summ_z_eta = []
                summ_x_tau = []
                summ_y_tau = []
                summ_z_tau = []
                for point in xyz:
                    index_of_nt = xyz.index(point)
                    summ_x_eta.append(point[0] * depsite[index_for_depsite][index_of_nt][0])
                    summ_y_eta.append(point[1] * depsite[index_for_depsite][index_of_nt][0])
                    summ_z_eta.append(point[2] * depsite[index_for_depsite][index_of_nt][0])
                    summ_x_tau.append(point[0] * depsite[index_for_depsite][index_of_nt][1])
                    summ_y_tau.append(point[1] * depsite[index_for_depsite][index_of_nt][1])
                    summ_z_tau.append(point[2] * depsite[index_for_depsite][index_of_nt][1])
                result.append([
                    [sum(summ_x_eta), sum(summ_x_tau)],
                    [sum(summ_y_eta), sum(summ_y_tau)],
                    [sum(summ_z_eta), sum(summ_z_tau)]
                ])
                index_for_depsite += 1
        return result

    def DEPSIxyzDEnt(self):
        result = []
        for eta in eta_for:
            for tau in tau_for:
                a = []
                for point in local_2d_points:
                    if local_2d_points.index(point) < 4:
                        a.append(self.PSINT_angel_main(eta, tau, point[0], point[1]))
                    elif local_2d_points.index(point) == 4 or local_2d_points.index(point) == 6:
                        a.append(self.PSINT_57_main(eta, tau, point[0], point[1]))
                    elif local_2d_points.index(point) == 5 or local_2d_points.index(point) == 7:
                        a.append(self.PSINT_68_main(eta, tau, point[0], point[1]))
                result.append(a)
        return result

    def FE_Calc(self, c_list, P_val, ZP_cast):
        DxyzDnt = self.DxyzDnt(ZP_cast)
        DEPSIxyzDEnt = self.DEPSIxyzDEnt()
        fe1 = []
        fe2 = []
        fe3 = []
        for i in range(8):  # [-1, -1]
            fe1_value = 0
            fe2_value = 0
            fe3_value = 0
            iterator_for_help = 0
            for m in c_list:
                for n in c_list:
                    DxyzDnt_item = DxyzDnt[iterator_for_help]
                    DEPSIxyzDEnt_item = DEPSIxyzDEnt[iterator_for_help][i]
                    fe1_value += m * n * P_val * (
                            DxyzDnt_item[1][0] * DxyzDnt_item[2][1] - DxyzDnt_item[2][0] * DxyzDnt_item[1][1]) \
                                 * DEPSIxyzDEnt_item
                    fe2_value += m * n * P_val * (
                            DxyzDnt_item[2][0] * DxyzDnt_item[0][1] - DxyzDnt_item[0][0] * DxyzDnt_item[2][1]) \
                                 * DEPSIxyzDEnt_item
                    fe3_value += m * n * P_val * (
                            DxyzDnt_item[0][0] * DxyzDnt_item[1][1] - DxyzDnt_item[1][0] * DxyzDnt_item[0][1]) \
                                 * DEPSIxyzDEnt_item
                    iterator_for_help += 1
            fe1.append(fe1_value)
            fe2.append(fe2_value)
            fe3.append(fe3_value)

        # Створюємо масив Fe розміром 60 і заповнюємо його нулями
        Fe = [0, 0, 0, 0, fe1[0], fe1[1], fe1[2], fe1[3], 0, 0,
              0, 0, 0, 0, 0, 0, fe1[4], fe1[5], fe1[6], fe1[7],
              0, 0, 0, 0, fe2[0], fe2[1], fe2[2], fe2[3], 0, 0,
              0, 0, 0, 0, 0, 0, fe2[4], fe2[5], fe2[6], fe2[7],
              0, 0, 0, 0, fe3[0], fe3[1], fe3[2], fe3[3], 0, 0,
              0, 0, 0, 0, 0, 0, fe3[4], fe3[5], fe3[6], fe3[7]]

        return Fe

    def MG_Create(self, All_MGE, AKT_RANGE, All_NT, ZU_cast):
        big_matrix = np.zeros((3 * AKT_RANGE, 3 * AKT_RANGE))
        result = big_matrix.tolist()

        for mge in All_MGE:
            index_of_MGE = All_MGE.index(mge)
            for j in range(60):
                for i in range(60):

                    if i < 20:
                        xyz_cord_i = 0
                        i_for_NT = i
                    elif 19 < i < 40:
                        xyz_cord_i = 1
                        i_for_NT = i - 20
                    else:
                        xyz_cord_i = 2
                        i_for_NT = i - 40

                    if j < 20:
                        xyz_cord_j = 0
                        j_for_NT = j
                    elif 19 < j < 40:
                        xyz_cord_j = 1
                        j_for_NT = j - 20
                    else:
                        xyz_cord_j = 2
                        j_for_NT = j - 40

                    index_i_for_MG = 3 * All_NT[index_of_MGE][i_for_NT] + xyz_cord_i
                    index_j_for_MG = 3 * All_NT[index_of_MGE][j_for_NT] + xyz_cord_j
                    result[index_j_for_MG][index_i_for_MG] += mge[j][i]

        for i in ZU_cast:
            index_of_point = ZU_cast.index(i)
            ix = 3 * index_of_point + 0
            iy = 3 * index_of_point + 1
            iz = 3 * index_of_point + 2
            result[ix][ix] = 10000000000000000
            result[iy][iy] = 10000000000000000
            result[iz][iz] = 10000000000000000

        return result

    def F_Create(self, All_Fe, AKT_RANGE, All_NT):
        big_matrix = np.zeros((3 * AKT_RANGE))
        result = big_matrix.tolist()
        for fe in All_Fe:
            index_of_FE = All_Fe.index(fe)
            for i in range(60):

                if i < 20:
                    xyz_cord_i = 0
                    i_for_NT = i
                elif 19 < i < 40:
                    xyz_cord_i = 1
                    i_for_NT = i - 20
                else:
                    xyz_cord_i = 2
                    i_for_NT = i - 40

                index_i_for_FE = 3 * All_NT[index_of_FE][i_for_NT] + xyz_cord_i
                result[index_i_for_FE] += fe[i]
        return result

    def on_all_points_button(self, event):
        # Отримання значень з полів введення
        a_val = float(self.a_entry.GetValue())
        b_val = float(self.b_entry.GetValue())
        c_val = float(self.c_entry.GetValue())
        na_val = int(self.n_A.GetValue())
        nb_val = int(self.n_B.GetValue())
        nc_val = int(self.n_C.GetValue())
        E = float(self.E_entry.GetValue())
        nu = float(self.nu_entry.GetValue())
        P = float(self.P_entry.GetValue())

        # a_val = 3
        # b_val = 2
        # c_val = 3
        # na_val = 3
        # nb_val = 2
        # nc_val = 3
        # E = 1
        # nu = 0.3
        # P = -0.2

        elements = []
        AKT = []
        NT = []
        # навантаження закріплених
        ZU = []
        # навантажений елемент
        ZP = []
        DFIABG = []
        DFIXYZ = []
        DJ = []
        liambda = E / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # Створення точок для прямокутника
        elements = self.create_points(a_val, b_val, c_val, na_val, nb_val, nc_val)
        AKT = self.separate_point(a_val, b_val, c_val, na_val, nb_val, nc_val)
        NT = self.NT_transform(AKT, elements)
        ZU = []
        ZP = []
        ZU_AKT = self.ZU_Chose(AKT)
        DFIABG = self.DFIABG_Create()
        for i in elements:
            DJ.append(self.DJ_Create(i))

        DJ_det = []
        for DJ_for_one in DJ:
            DJ_det_for_one = []
            for i in DJ_for_one:
                DJ_det_for_one.append(self.calculate_determinant(i))
            DJ_det.append(DJ_det_for_one)

        DFIXYZ = self.Solv_SLAR_for_elements(elements, DJ, DFIABG)

        list_of_MGE = []
        for i in range(len(elements)):
            list_of_MGE.append(
                self.calc_MGE(DFIXYZ[i], DJ_det[i], [c_1, c_2, c_3], liambda, nu, mu))

        for i in AKT:
            if i[2] == 0:
                ZU.append(i)

        ZP.append(self.ZP_Chose(elements[2], 6, 2))
        # ZP.append(self.ZP_Chose(elements[3], 6, 2))

        for i in range(len(NT) - len(ZP)):
            FE.append(np.zeros(60).tolist())

        for i in ZP:
            FE.append(self.FE_Calc([c_1, c_2, c_3], P, i))

        MG = self.MG_Create(list_of_MGE, len(AKT), NT, ZU)

        F = self.F_Create(FE, len(AKT), NT)
        result_points = self.solve_linear_equation(MG, F)

        x_points = [sublist[0] for sublist in AKT]
        y_points = [sublist[1] for sublist in AKT]
        z_points = [sublist[2] for sublist in AKT]
        # Додавання точок до графіку
        x_points_modified = np.zeros(len(AKT)).tolist()
        y_points_modified = np.zeros(len(AKT)).tolist()
        z_points_modified = np.zeros(len(AKT)).tolist()
        # Створення фігури та 3D-простору

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.scatter(x_points, y_points, z_points, c='r', marker='o')

        # Встановлюємо пропорції осей
        self.ax.set_box_aspect([a_val, b_val, c_val])

        self.fig.canvas.draw()

        # Оновлення відображення графіку
        for i in result_points:
            index_of_point = result_points.index(i)
            j = index_of_point // 3
            if (index_of_point + 1) % 3 == 1:
                x_points_modified[j] = x_points[j] + i
                j += 1
            if (index_of_point + 1) % 3 == 2:
                y_points_modified[j] = y_points[j] + i
                j += 1
            if (index_of_point + 1) % 3 == 0:
                z_points_modified[j] = z_points[j] + i
                j += 1

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.scatter(x_points_modified, y_points_modified, z_points_modified, c='g', marker='o')

        # Встановлюємо пропорції осей
        self.ax.set_box_aspect([a_val, b_val, c_val])

        self.fig.canvas.draw()

        # Оновлення відображення графіку
        plt.show()
        a = 0


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Main App", size=wx.Size(200, 800))
        panel = MyPanel(self)


if __name__ == "__main__":
    local_points = [
        [-1, 1, -1],  # 1
        [1, 1, -1],  # 2
        [1, -1, -1],  # 3
        [-1, -1, -1],  # 4
        [-1, 1, 1],  # 5
        [1, 1, 1],  # 6
        [1, -1, 1],  # 7
        [-1, -1, 1],  # 8
        [0, 1, -1],  # 9
        [1, 0, -1],  # 10
        [0, -1, -1],  # 11
        [-1, 0, -1],  # 12
        [-1, 1, 0],  # 13
        [1, 1, 0],  # 14
        [1, -1, 0],  # 15
        [-1, -1, 0],  # 16
        [0, 1, 1],  # 17
        [1, 0, 1],  # 18
        [0, -1, 1],  # 19
        [-1, 0, 1]  # 20
    ]
    # en\tau
    local_2d_points = [
        [-1, -1],  # 1
        [1, -1],  # 2
        [1, 1],  # 3
        [-1, 1],  # 4
        [0, -1],  # 5
        [1, 0],  # 6
        [0, 1],  # 7
        [-1, 0]  # 8
    ]

    elements = []
    AKT = []
    NT = []
    # навантаження закріплених
    ZU = []
    # навантажений елемент
    ZP = []
    DFIABG = []
    DFIXYZ = []
    DJ = []
    FE = []
    F = []
    MG = []
    nol_shist = math.sqrt(0.6)
    alpha_for = [-nol_shist, 0, nol_shist]
    beta_for = [-nol_shist, 0, nol_shist]
    gamma_for = [-nol_shist, 0, nol_shist]
    eta_for = [-nol_shist, 0, nol_shist]
    tau_for = [-nol_shist, 0, nol_shist]

    c_1 = 5 / 9
    c_2 = 8 / 9
    c_3 = 5 / 9

    E = 0
    nu = 0
    P = 0

    liambda = 0
    mu = 0

    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()
