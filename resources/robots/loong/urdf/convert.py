import mujoco

# 修改为你实际的文件名
input_urdf = "loong.urdf"
output_xml = "loong.xml"

try:
    # 加载模型
    model = mujoco.MjModel.from_xml_path(input_urdf)
    # 保存为 XML
    mujoco.mj_saveLastXML(output_xml, model)
    print(f"成功！已生成 {output_xml}")
except Exception as e:
    print(f"转换失败: {e}")