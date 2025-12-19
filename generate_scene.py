import os
import mujoco

def generate_scene(xml_path: str, out_name: str = "scene.xml"):
    """
    Load an MJCF that uses <replicate>, let MuJoCo expand it,
    and save the expanded scene.xml next to the original xml.
    """
    xml_path = os.path.abspath(xml_path)
    xml_dir = os.path.dirname(xml_path)
    out_path = os.path.join(xml_dir, out_name)

    # 1. Load model (replicate is expanded internally here)
    model = mujoco.MjModel.from_xml_path(xml_path)

    # 2. Save expanded XML (replicate tags are gone)
    mujoco.mj_saveLastXML(out_path, model)

    print(f"[OK] Expanded scene saved to:\n  {out_path}")


if __name__ == "__main__":
    # 假设脚本和 ubot_ax_centered.xml 在同一目录
    generate_scene("ubot_ax_centered.xml")
