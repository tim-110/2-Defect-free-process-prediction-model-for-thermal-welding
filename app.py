from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
import logging
from welding_model import WeldingAnalysisSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 初始化分析系统
system = WeldingAnalysisSystem()
is_trained = False


@app.route('/')
def index():
    """首页"""
    return jsonify({
        'status': 'success',
        'message': '智能焊接质量控制系统 API',
        'version': '1.0.0'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'success',
        'message': '系统运行正常',
        'model_trained': is_trained
    })


@app.route('/api/train-models', methods=['POST'])
def train_models():
    """训练焊接模型"""
    global is_trained
    try:
        logger.info("开始训练焊接模型...")

        # 训练模型
        system.train_models()
        is_trained = True

        logger.info("模型训练完成")
        return jsonify({
            'status': 'success',
            'message': '模型训练完成',
            'arc_r2': float(system.arc_welding_model.r_squared),
            'laser_r2': float(system.laser_welding_model.r_squared)
        })

    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'模型训练失败: {str(e)}'
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_defects():
    """预测缺陷概率"""
    try:
        data = request.json
        welding_type = data.get('welding_type', 'arc')
        parameters = data.get('parameters', [])

        if not parameters:
            return jsonify({
                'status': 'error',
                'message': '参数不能为空'
            }), 400

        # 设置焊接类型
        system.set_welding_type(welding_type)

        # 预测缺陷概率
        prediction = system.predict_defects(np.array([parameters]))

        # 获取风险等级
        crack_risk = '低风险' if prediction[0][0] < 1 else '中风险' if prediction[0][0] < 2 else '高风险'
        porosity_risk = '低风险' if prediction[0][1] < 1 else '中风险' if prediction[0][1] < 2 else '高风险'

        return jsonify({
            'status': 'success',
            'prediction': {
                'crack_probability': float(prediction[0][0]),
                'porosity_probability': float(prediction[0][1]),
                'crack_risk': crack_risk,
                'porosity_risk': porosity_risk
            },
            'welding_type': welding_type
        })

    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'预测失败: {str(e)}'
        }), 500


@app.route('/api/zero-crack-region', methods=['POST'])
def find_zero_crack_region():
    """寻找零裂纹区间"""
    try:
        data = request.json
        welding_type = data.get('welding_type', 'arc')
        max_prob = data.get('max_prob', 0.1)

        system.set_welding_type(welding_type)
        ranges, samples = system.find_zero_crack_region(max_prob)

        if ranges is None:
            return jsonify({
                'status': 'success',
                'message': '未找到满足条件的零裂纹区间',
                'ranges': {},
                'sample_count': 0
            })

        # 转换numpy类型为Python原生类型
        result_ranges = {}
        for param, (min_val, max_val) in ranges.items():
            result_ranges[param] = {
                'min': float(min_val),
                'max': float(max_val)
            }

        return jsonify({
            'status': 'success',
            'ranges': result_ranges,
            'sample_count': len(samples),
            'max_defect_prob': max_prob
        })

    except Exception as e:
        logger.error(f"寻找零裂纹区间失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'寻找零裂纹区间失败: {str(e)}'
        }), 500


@app.route('/api/3d-surface', methods=['POST'])
def generate_3d_surface():
    """生成3D响应曲面数据"""
    try:
        data = request.json
        welding_type = data.get('welding_type', 'arc')
        param1 = data.get('param1', 'current')
        param2 = data.get('param2', 'voltage')
        defect_type = data.get('defect_type', 0)

        system.set_welding_type(welding_type)

        # 生成网格数据
        param_ranges = system.current_model.param_ranges
        x_range = np.linspace(param_ranges[param1][0], param_ranges[param1][1], 20)
        y_range = np.linspace(param_ranges[param2][0], param_ranges[param2][1], 20)

        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros(X.shape)

        # 计算响应面
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                params = system._get_default_parameters()
                param_names = list(param_ranges.keys())
                params[param_names.index(param1)] = X[i, j]
                params[param_names.index(param2)] = Y[i, j]

                prediction = system.current_model.predict(np.array([params]))
                Z[i, j] = prediction[0]

        # 转换为列表格式
        surface_data = {
            'x': X.tolist(),
            'y': Y.tolist(),
            'z': Z.tolist(),
            'param1': param1,
            'param2': param2,
            'defect_type': defect_type
        }

        return jsonify({
            'status': 'success',
            'surface_data': surface_data
        })

    except Exception as e:
        logger.error(f"生成3D曲面失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'生成3D曲面失败: {str(e)}'
        }), 500


@app.route('/api/parameter-slice', methods=['POST'])
def generate_parameter_slice():
    """生成参数切片数据"""
    try:
        data = request.json
        welding_type = data.get('welding_type', 'arc')
        fixed_param = data.get('fixed_param', 'current')
        fixed_value = data.get('fixed_value', 220)
        varying_param = data.get('varying_param', 'voltage')

        system.set_welding_type(welding_type)

        # 生成切片数据
        param_ranges = system.current_model.param_ranges
        varying_range = np.linspace(param_ranges[varying_param][0], param_ranges[varying_param][1], 50)

        defect_probs = []
        for value in varying_range:
            params = system._get_default_parameters()
            param_names = list(param_ranges.keys())
            params[param_names.index(fixed_param)] = fixed_value
            params[param_names.index(varying_param)] = value

            prediction = system.current_model.predict(np.array([params]))
            defect_probs.append(float(prediction[0]))

        slice_data = {
            'varying_param': varying_param,
            'varying_range': varying_range.tolist(),
            'defect_probs': defect_probs,
            'fixed_param': fixed_param,
            'fixed_value': fixed_value
        }

        return jsonify({
            'status': 'success',
            'slice_data': slice_data
        })

    except Exception as e:
        logger.error(f"生成参数切片失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'生成参数切片失败: {str(e)}'
        }), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    try:
        arc_info = {
            'welding_type': '电弧焊',
            'parameters': list(system.arc_welding_model.param_ranges.keys()),
            'r_squared': float(system.arc_welding_model.r_squared) if system.arc_welding_model.r_squared else 0,
            'trained': system.arc_welding_model.model is not None
        }

        laser_info = {
            'welding_type': '激光焊',
            'parameters': list(system.laser_welding_model.param_ranges.keys()),
            'r_squared': float(system.laser_welding_model.r_squared) if system.laser_welding_model.r_squared else 0,
            'trained': system.laser_welding_model.model is not None
        }

        return jsonify({
            'status': 'success',
            'models': {
                'arc_welding': arc_info,
                'laser_welding': laser_info
            },
            'system_trained': is_trained
        })

    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'获取模型信息失败: {str(e)}'
        }), 500


@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """获取样本数据"""
    try:
        # 生成电弧焊样本数据
        X_arc, y_arc = system.arc_welding_model.generate_sample_data(30)
        arc_data = []
        for i in range(len(X_arc)):
            arc_data.append({
                'current': float(X_arc[i][0]),
                'voltage': float(X_arc[i][1]),
                'speed': float(X_arc[i][2]),
                'gas_flow': float(X_arc[i][3]),
                'crack_prob': float(y_arc[i][0]),
                'porosity_prob': float(y_arc[i][1])
            })

        # 生成激光焊样本数据
        X_laser, y_laser = system.laser_welding_model.generate_sample_data(30)
        laser_data = []
        for i in range(len(X_laser)):
            laser_data.append({
                'power': float(X_laser[i][0]),
                'spot_diameter': float(X_laser[i][1]),
                'defocus': float(X_laser[i][2]),
                'scan_speed': float(X_laser[i][3]),
                'lack_penetration_prob': float(y_laser[i][0]),
                'hot_crack_prob': float(y_laser[i][1])
            })

        return jsonify({
            'status': 'success',
            'arc_welding': arc_data,
            'laser_welding': laser_data
        })

    except Exception as e:
        logger.error(f"获取样本数据失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'获取样本数据失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    logger.info("启动智能焊接质量控制系统API服务...")
    app.run(debug=True, host='0.0.0.0', port=5000)