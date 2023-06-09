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

    def ZP_Chose(self, eleme):
        maxim = max([sublist[2] for sublist in eleme])
        return [sublist for sublist in eleme if sublist[2] == maxim]

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
            (alpha_i * (2 * alpha * alpha_i + beta * beta_i + gamma * gamma_i - 1)),

            (1 / 8) * (1 + alpha * alpha_i) * (1 + gamma * gamma_i) *
            (beta_i * (alpha * alpha_i + 2 * beta * beta_i + gamma * gamma_i - 1)),

            (1 / 8) * (1 + beta * beta_i) * (1 + alpha * alpha_i) *
            (gamma_i * (alpha * alpha_i + beta * beta_i + 2 * gamma * gamma_i - 1))
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
        result = []
        for gamma in gamma_for:
            for beta in beta_for:
                for alpha in alpha_for:
                    result.append(self.DELTA(xyz, alpha, beta, gamma))
        return result

    def DELTA(self, xyz, alpha, beta, gamma):
        result = [
            self.DxyzDabg(xyz, alpha, beta, gamma, 0),
            self.DxyzDabg(xyz, alpha, beta, gamma, 1),
            self.DxyzDabg(xyz, alpha, beta, gamma, 2)
        ]
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
        for i in elements_cast:
            result.append(self.Solv_SLAR_for_element(i, DJ_cast, DFIABG_cast))
        return result

    def Solv_SLAR_for_element(self, elements_cast, DJacobian, DFIABG_cast):
        result = []
        for delta_item in DJacobian:
            index_of_item = DJacobian.index(delta_item)
            dfixyz = []
            for points in DFIABG_cast[index_of_item]:
                solved_L_E = self.solve_linear_equation(delta_item, points)
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
        help_array = []
        for i in range(20):
            line_of_matrix_a11 = []
            line_of_matrix_a22 = []
            line_of_matrix_a33 = []
            line_of_matrix_a12 = []
            line_of_matrix_a13 = []
            line_of_matrix_a23 = []
            for j in range(20):
                a11 = 0
                a22 = 0
                a33 = 0
                a12 = 0
                a13 = 0
                a23 = 0
                general_index = 0
                for m in c_list:
                    for n in c_list:
                        for k in c_list:
                            dfi = DFIXYZ_cast[general_index]

                            help_item = m * n * k * (lambda_val * (1 - nu_val) * dfi[i][0] * dfi[j][0] +
                                                     mu_val * (dfi[i][1] * dfi[j][1] + dfi[i][2] * dfi[j][2])) \
                                        * determinant_list[general_index]
                            help_array.append(help_item)
                            a11 += help_item

                            a22 += m * n * k * (lambda_val * (1 - nu_val) * dfi[i][1] * dfi[j][1] +
                                                mu_val * (dfi[i][0] * dfi[j][0] + dfi[i][2] * dfi[j][2])) \
                                   * determinant_list[general_index]

                            a33 += m * n * k * (lambda_val * (1 - nu_val) * dfi[i][2] * dfi[j][2] +
                                                mu_val * (dfi[i][0] * dfi[j][0] + dfi[i][1] * dfi[j][1])) \
                                   * determinant_list[general_index]

                            a12 += m * n * k * (lambda_val * nu_val * dfi[i][0] * dfi[j][1] +
                                                mu_val * dfi[i][1] * dfi[j][0]) * determinant_list[general_index]

                            a13 += m * n * k * (lambda_val * nu_val * dfi[i][0] * dfi[j][2] +
                                                mu_val * dfi[i][2] * dfi[j][0]) * determinant_list[general_index]

                            a23 += m * n * k * (lambda_val * nu_val * dfi[i][1] * dfi[j][2] +
                                                mu_val * dfi[i][2] * dfi[j][1]) * determinant_list[general_index]
                            general_index += 1
                line_of_matrix_a11.append(a11)
                line_of_matrix_a22.append(a22)
                line_of_matrix_a33.append(a33)
                line_of_matrix_a12.append(a12)
                line_of_matrix_a13.append(a13)
                line_of_matrix_a23.append(a23)
            matrix_a11.append(line_of_matrix_a11)
            matrix_a22.append(line_of_matrix_a22)
            matrix_a33.append(line_of_matrix_a33)
            matrix_a12.append(line_of_matrix_a12)
            matrix_a13.append(line_of_matrix_a13)
            matrix_a23.append(line_of_matrix_a23)

        matrix1 = np.array(matrix_a11)
        matrix2 = np.array(matrix_a12)
        matrix3 = np.array(matrix_a13)
        matrix4 = np.array(matrix_a22)
        matrix5 = np.array(matrix_a23)
        matrix6 = np.array(matrix_a33)

        # Створіть нову матрицю 60 на 60, заповнену нулями
        big_matrix = np.zeros((60, 60))

        # Заповніть велику матрицю зі шести матриць
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

        liambda = E / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # Створення точок для прямокутника
        elements = self.create_points(a_val, b_val, c_val, na_val, nb_val, nc_val)
        AKT = self.separate_point(a_val, b_val, c_val, na_val, nb_val, nc_val)
        NT = self.NT_transform(AKT, elements)

        ZU = self.ZU_Chose(AKT)
        ZP = self.ZP_Chose(AKT)
        DFIABG = self.DFIABG_Create()
        DJ = self.DJ_Create(elements[0])
        DJ_det = []
        for i in DJ:
            DJ_det.append(self.calculate_determinant(i))

        DFIXYZ = self.Solv_SLAR_for_elements(elements, DJ, DFIABG)

        list_of_MGE = []
        for i in elements:
            index_of_list = elements.index(i)
            list_of_MGE.append(self.calc_MGE(DFIXYZ[index_of_list], DJ_det, [c_1, c_2, c_3], liambda, nu, mu))

        x_points = [sublist[0] for sublist in AKT]
        y_points = [sublist[1] for sublist in AKT]
        z_points = [sublist[2] for sublist in AKT]
        # Додавання точок до графіку

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
        plt.show()


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Main App", size=wx.Size(1000, 700))
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

    nol_shist = math.sqrt(0.6)
    alpha_for = [-nol_shist, 0, nol_shist]
    beta_for = [-nol_shist, 0, nol_shist]
    gamma_for = [-nol_shist, 0, nol_shist]

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
