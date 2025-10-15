import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from scipy import optimize
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class WeldingResponseSurfaceModel:
    """焊接响应面回归模型基类"""

    def __init__(self, welding_type):
        self.welding_type = welding_type
        self.model = None
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_names = []
        self.coefficients = None
        self.r_squared = None

    def preprocess_data(self, X, y):
        """数据预处理"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 生成多项式特征
        X_poly = self.poly.fit_transform(X_scaled)

        # 获取特征名称
        self.feature_names = self.poly.get_feature_names_out([f'x{i + 1}' for i in range(X.shape[1])])

        return X_poly, y

    def train_model(self, X, y):
        """训练响应面回归模型"""
        X_poly, y = self.preprocess_data(X, y)

        # 使用statsmodels进行回归分析
        X_poly = sm.add_constant(X_poly)
        model = sm.OLS(y, X_poly).fit()

        self.model = model
        self.coefficients = model.params
        self.r_squared = model.rsquared

        print(f"{self.welding_type}模型训练完成")
        print(f"R²: {self.r_squared:.4f}")
        print(model.summary())

        return model

    def predict(self, X):
        """预测缺陷概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        X_scaled = self.scaler.transform(X)
        X_poly = self.poly.transform(X_scaled)
        X_poly = sm.add_constant(X_poly, has_constant='add')

        return self.model.predict(X_poly)

    def get_optimal_parameters(self, objective_function, bounds, constraints=None):
        """获取最优工艺参数"""
        result = optimize.minimize(objective_function,
                                   x0=np.mean(bounds, axis=1),
                                   bounds=bounds,
                                   constraints=constraints,
                                   method='SLSQP')

        return result


class ArcWeldingModel(WeldingResponseSurfaceModel):
    """电弧焊响应面回归模型"""

    def __init__(self):
        super().__init__("电弧焊")
        self.param_ranges = {
            'current': (180, 250),  # 焊接电流 (A)
            'voltage': (24, 30),  # 焊接电压 (V)
            'speed': (25, 40),  # 焊接速度 (cm/min)
            'gas_flow': (15, 25)  # 保护气体流量 (L/min)
        }

    def preprocess_data(self, X, y):
        """电弧焊数据预处理"""
        # 电弧稳定性筛选：剔除电流波动 > 5% 的异常数据
        if len(X) > 0:
            current_std = np.std(X[:, 0])
            current_mean = np.mean(X[:, 0])
            if current_mean > 0:
                valid_indices = np.where(np.abs(X[:, 0] - current_mean) / current_mean <= 0.05)[0]
                X = X[valid_indices]
                y = y[valid_indices]

        return super().preprocess_data(X, y)

    def generate_sample_data(self, n_samples=50):
        """生成电弧焊样本数据"""
        np.random.seed(42)

        # 生成参数
        current = np.random.uniform(180, 250, n_samples)
        voltage = np.random.uniform(24, 30, n_samples)
        speed = np.random.uniform(25, 40, n_samples)
        gas_flow = np.random.uniform(15, 25, n_samples)

        X = np.column_stack([current, voltage, speed, gas_flow])

        # 基于真实物理关系生成缺陷概率
        crack_prob = (0.01 * (current - 200) ** 2 +
                      0.02 * (voltage - 27) ** 2 +
                      0.03 * (speed - 32) ** 2 +
                      0.01 * (gas_flow - 20) ** 2 +
                      np.random.normal(0, 0.5, n_samples))
        crack_prob = np.clip(crack_prob, 0, 10)

        porosity_prob = (0.02 * (current - 220) ** 2 +
                         0.01 * (voltage - 28) ** 2 +
                         0.04 * (speed - 30) ** 2 +
                         0.03 * (gas_flow - 18) ** 2 +
                         np.random.normal(0, 0.3, n_samples))
        porosity_prob = np.clip(porosity_prob, 0, 8)

        y = np.column_stack([crack_prob, porosity_prob])

        return X, y


class LaserWeldingModel(WeldingResponseSurfaceModel):
    """激光焊响应面回归模型"""

    def __init__(self):
        super().__init__("激光焊")
        self.param_ranges = {
            'power': (1500, 2500),  # 激光功率 (W)
            'spot_diameter': (0.2, 0.8),  # 光斑直径 (mm)
            'defocus': (-3, 5),  # 离焦量 (mm)
            'scan_speed': (300, 600)  # 扫描速度 (mm/s)
        }

    def preprocess_data(self, X, y):
        """激光焊数据预处理"""
        if len(X) > 0:
            # 计算激光能量密度并筛选有效数据
            power = X[:, 0]
            spot_diameter = X[:, 1]
            scan_speed = X[:, 3]

            energy_density = power / (np.pi * (spot_diameter / 2) ** 2 * scan_speed)
            valid_indices = np.where((energy_density >= 80) & (energy_density <= 200))[0]

            X = X[valid_indices]
            y = y[valid_indices]

        return super().preprocess_data(X, y)

    def generate_sample_data(self, n_samples=50):
        """生成激光焊样本数据"""
        np.random.seed(42)

        # 生成参数
        power = np.random.uniform(1500, 2500, n_samples)
        spot_diameter = np.random.uniform(0.2, 0.8, n_samples)
        defocus = np.random.uniform(-3, 5, n_samples)
        scan_speed = np.random.uniform(300, 600, n_samples)

        X = np.column_stack([power, spot_diameter, defocus, scan_speed])

        # 基于真实物理关系生成缺陷概率
        lack_penetration = (0.0001 * (power - 2000) ** 2 +
                            0.5 * (spot_diameter - 0.5) ** 2 +
                            0.1 * (defocus - 1) ** 2 +
                            0.0001 * (scan_speed - 450) ** 2 +
                            np.random.normal(0, 0.2, n_samples))
        lack_penetration = np.clip(lack_penetration, 0, 5)

        hot_crack_prob = (0.0002 * (power - 2200) ** 2 +
                          0.3 * (spot_diameter - 0.6) ** 2 +
                          0.05 * (defocus - 2) ** 2 +
                          0.00005 * (scan_speed - 500) ** 2 +
                          np.random.normal(0, 0.3, n_samples))
        hot_crack_prob = np.clip(hot_crack_prob, 0, 6)

        y = np.column_stack([lack_penetration, hot_crack_prob])

        return X, y


class WeldingAnalysisSystem:
    """焊接分析系统主类"""

    def __init__(self):
        self.arc_welding_model = ArcWeldingModel()
        self.laser_welding_model = LaserWeldingModel()
        self.current_model = None

    def train_models(self):
        """训练两种焊接模型"""
        print("开始训练电弧焊模型...")
        X_arc, y_arc = self.arc_welding_model.generate_sample_data()
        self.arc_welding_model.train_model(X_arc, y_arc[:, 0])  # 训练裂纹概率模型

        print("\n开始训练激光焊模型...")
        X_laser, y_laser = self.laser_welding_model.generate_sample_data()
        self.laser_welding_model.train_model(X_laser, y_laser[:, 0])  # 训练未熔透概率模型

    def set_welding_type(self, welding_type):
        """设置当前焊接类型"""
        if welding_type == "arc" or welding_type == "电弧焊":
            self.current_model = self.arc_welding_model
        elif welding_type == "laser" or welding_type == "激光焊":
            self.current_model = self.laser_welding_model
        else:
            raise ValueError("不支持的焊接类型")

    def predict_defects(self, parameters):
        """预测缺陷概率"""
        if self.current_model is None:
            raise ValueError("请先设置焊接类型")

        return self.current_model.predict(np.array([parameters]))

    def create_3d_surface(self, param1, param2, defect_type=0):
        """创建3D响应曲面"""
        if self.current_model is None:
            raise ValueError("请先设置焊接类型")

        # 生成网格数据
        x_range = np.linspace(*self.current_model.param_ranges[param1], 20)
        y_range = np.linspace(*self.current_model.param_ranges[param2], 20)
        X, Y = np.meshgrid(x_range, y_range)

        # 计算Z值（缺陷概率）
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # 构建完整的参数向量
                params = self._get_default_parameters()
                param_names = list(self.current_model.param_ranges.keys())
                params[param_names.index(param1)] = X[i, j]
                params[param_names.index(param2)] = Y[i, j]

                Z[i, j] = self.current_model.predict(np.array([params]))[0]

        return X, Y, Z

    def create_parameter_slice(self, fixed_param, fixed_value, varying_param, defect_type=0):
        """创建参数切片图"""
        if self.current_model is None:
            raise ValueError("请先设置焊接类型")

        # 生成变化参数的范围
        varying_range = np.linspace(*self.current_model.param_ranges[varying_param], 50)

        # 计算缺陷概率
        defect_probs = []
        for value in varying_range:
            params = self._get_default_parameters()
            param_names = list(self.current_model.param_ranges.keys())
            params[param_names.index(fixed_param)] = fixed_value
            params[param_names.index(varying_param)] = value

            defect_probs.append(self.current_model.predict(np.array([params]))[0])

        return varying_range, defect_probs

    def find_zero_crack_region(self, max_defect_prob=0.1):
        """寻找零裂纹工艺区间"""
        if self.current_model is None:
            raise ValueError("请先设置焊接类型")

        # 生成参数网格
        param_names = list(self.current_model.param_ranges.keys())
        n_points = 10000  # 蒙特卡洛采样点数

        # 随机采样参数空间
        samples = []
        for param_name in param_names:
            samples.append(np.random.uniform(*self.current_model.param_ranges[param_name], n_points))

        samples = np.column_stack(samples)

        # 预测缺陷概率
        defect_probs = self.current_model.predict(samples)

        # 筛选满足条件的参数组合
        valid_indices = np.where(defect_probs <= max_defect_prob)[0]
        valid_samples = samples[valid_indices]

        if len(valid_samples) == 0:
            print("未找到满足条件的参数组合")
            return None, []

        # 计算各参数的区间范围
        param_ranges = {}
        for i, param_name in enumerate(param_names):
            param_values = valid_samples[:, i]
            param_ranges[param_name] = (np.min(param_values), np.max(param_values))

        return param_ranges, valid_samples

    def _get_default_parameters(self):
        """获取默认参数值（各参数范围的中点）"""
        if self.current_model is None:
            raise ValueError("请先设置焊接类型")

        default_params = []
        for param_range in self.current_model.param_ranges.values():
            default_params.append(np.mean(param_range))

        return default_params


# 使用示例
if __name__ == "__main__":
    # 创建分析系统
    system = WeldingAnalysisSystem()

    # 训练模型
    system.train_models()

    # 设置焊接类型为电弧焊
    system.set_welding_type("电弧焊")

    # 测试预测
    test_params = [220, 28, 32, 20]  # 电流, 电压, 速度, 气体流量
    prediction = system.predict_defects(test_params)
    print(f"预测结果: {prediction}")

    # 寻找零裂纹区间
    zero_crack_ranges, valid_samples = system.find_zero_crack_region()
    if zero_crack_ranges:
        print("零裂纹工艺区间:")
        for param, (min_val, max_val) in zero_crack_ranges.items():
            print(f"{param}: {min_val:.2f} - {max_val:.2f}")