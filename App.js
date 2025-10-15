import React, { useState, useEffect } from 'react';
import { Layout, Card, Tabs, Alert, Button, Spin, Row, Col } from 'antd';
import WeldingPredictionForm from './components/WeldingPredictionForm';
import ZeroCrackAnalysis from './components/ZeroCrackAnalysis';
import { useWeldingAnalysis } from './hooks/useWeldingAnalysis';
import 'antd/dist/reset.css';
import './App.css';

const { Header, Content } = Layout;
const { TabPane } = Tabs;

function App() {
  const [activeTab, setActiveTab] = useState('prediction');
  const [apiStatus, setApiStatus] = useState('checking');
  const { loading, modelInfo, fetchModelInfo, healthCheck, trainModels } = useWeldingAnalysis();

  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    const isHealthy = await healthCheck();
    setApiStatus(isHealthy ? 'healthy' : 'error');

    if (isHealthy) {
      await fetchModelInfo();
    }
  };

  const handleTrainModels = async () => {
    await trainModels();
    await fetchModelInfo();
  };

  const renderStatusAlert = () => {
    if (apiStatus === 'checking') {
      return <Alert message="检查后端服务状态..." type="info" showIcon />;
    }

    if (apiStatus === 'error') {
      return (
        <Alert
          message="后端服务连接失败"
          description="请确保Python Flask服务正在运行在 http://localhost:5000"
          type="error"
          showIcon
          action={
            <Button size="small" onClick={checkApiStatus}>
              重试
            </Button>
          }
        />
      );
    }

    return (
      <Alert
        message="系统状态正常"
        description={`模型训练状态: ${modelInfo?.system_trained ? '已训练' : '未训练'}`}
        type="success"
        showIcon
        action={
          !modelInfo?.system_trained && (
            <Button size="small" type="primary" onClick={handleTrainModels} loading={loading}>
              训练模型
            </Button>
          )
        }
      />
    );
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', color: 'white', padding: '0 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
          <h1 style={{ color: 'white', margin: 0, fontSize: '20px' }}>
            🔥 智能焊接质量控制系统
          </h1>
        </div>
      </Header>

      <Content style={{ padding: '24px' }}>
        {renderStatusAlert()}

        <Card style={{ marginTop: '24px' }}>
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            <TabPane tab="缺陷预测" key="prediction">
              <WeldingPredictionForm />
            </TabPane>
            <TabPane tab="零裂纹分析" key="analysis">
              <ZeroCrackAnalysis />
            </TabPane>
            <TabPane tab="系统状态" key="dashboard">
              <Spin spinning={loading}>
                {modelInfo && (
                  <div>
                    <Tabs defaultActiveKey="arc">
                      <TabPane tab="电弧焊模型" key="arc">
                        <Card>
                          <Row gutter={16}>
                            <Col span={8}>
                              <p><strong>训练状态:</strong></p>
                              <p>{modelInfo.models.arc_welding.trained ? '✅ 已训练' : '❌ 未训练'}</p>
                            </Col>
                            <Col span={8}>
                              <p><strong>R²得分:</strong></p>
                              <p>{modelInfo.models.arc_welding.r_squared.toFixed(4)}</p>
                            </Col>
                            <Col span={8}>
                              <p><strong>参数:</strong></p>
                              <p>{modelInfo.models.arc_welding.parameters.join(', ')}</p>
                            </Col>
                          </Row>
                        </Card>
                      </TabPane>
                      <TabPane tab="激光焊模型" key="laser">
                        <Card>
                          <Row gutter={16}>
                            <Col span={8}>
                              <p><strong>训练状态:</strong></p>
                              <p>{modelInfo.models.laser_welding.trained ? '✅ 已训练' : '❌ 未训练'}</p>
                            </Col>
                            <Col span={8}>
                              <p><strong>R²得分:</strong></p>
                              <p>{modelInfo.models.laser_welding.r_squared.toFixed(4)}</p>
                            </Col>
                            <Col span={8}>
                              <p><strong>参数:</strong></p>
                              <p>{modelInfo.models.laser_welding.parameters.join(', ')}</p>
                            </Col>
                          </Row>
                        </Card>
                      </TabPane>
                    </Tabs>
                  </div>
                )}
              </Spin>
            </TabPane>
          </Tabs>
        </Card>
      </Content>
    </Layout>
  );
}

export default App;