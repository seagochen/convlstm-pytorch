"""
类别配置模块

用于管理可扩展的分类类别，支持从二分类到多分类的灵活配置。
"""

from typing import List, Optional, Dict


# 默认类别配置
DEFAULT_CLASSES = ['static', 'dynamic', 'negative']


class ClassConfig:
    """
    类别配置管理器

    用于统一管理分类任务的类别名称和标签映射。

    Args:
        class_names: 类别名称列表，按标签顺序排列
                     例如: ['static', 'dynamic', 'negative']
                     标签值为列表索引: static=0, dynamic=1, negative=2

    Example:
        # 三分类（默认）
        config = ClassConfig()

        # 自定义类别
        config = ClassConfig(['background', 'fire', 'smoke', 'explosion'])

        # 二分类兼容模式
        config = ClassConfig(['static', 'dynamic'])
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        if class_names is None:
            class_names = DEFAULT_CLASSES.copy()

        if len(class_names) < 2:
            raise ValueError("至少需要 2 个类别")

        # 检查类别名称唯一性
        if len(class_names) != len(set(class_names)):
            raise ValueError("类别名称必须唯一")

        self._class_names = class_names
        self._name_to_label = {name: idx for idx, name in enumerate(class_names)}
        self._label_to_name = {idx: name for idx, name in enumerate(class_names)}

    @property
    def class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self._class_names.copy()

    @property
    def num_classes(self) -> int:
        """获取类别数量"""
        return len(self._class_names)

    def label_to_name(self, label: int) -> str:
        """
        将标签值转换为类别名称

        Args:
            label: 标签值 (0, 1, 2, ...)

        Returns:
            类别名称
        """
        if label not in self._label_to_name:
            raise ValueError(f"未知标签: {label}, 有效范围: 0-{self.num_classes-1}")
        return self._label_to_name[label]

    def name_to_label(self, name: str) -> int:
        """
        将类别名称转换为标签值

        Args:
            name: 类别名称

        Returns:
            标签值
        """
        if name not in self._name_to_label:
            raise ValueError(f"未知类别: {name}, 有效类别: {self._class_names}")
        return self._name_to_label[name]

    def has_class(self, name: str) -> bool:
        """检查是否包含指定类别"""
        return name in self._name_to_label

    def get_label_safe(self, name: str, default: int = -1) -> int:
        """
        安全地获取标签值，未知类别返回默认值

        Args:
            name: 类别名称
            default: 未知类别时的默认返回值

        Returns:
            标签值或默认值
        """
        return self._name_to_label.get(name, default)

    def get_statistics_template(self) -> Dict[str, int]:
        """获取用于统计的类别计数模板"""
        return {name: 0 for name in self._class_names}

    def __repr__(self) -> str:
        return f"ClassConfig({self._class_names})"

    def __str__(self) -> str:
        class_str = ", ".join([f"{name}={idx}" for idx, name in enumerate(self._class_names)])
        return f"ClassConfig: {self.num_classes} classes [{class_str}]"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ClassConfig):
            return False
        return self._class_names == other._class_names

    def to_dict(self) -> Dict:
        """序列化为字典（用于保存配置）"""
        return {
            'class_names': self._class_names,
            'num_classes': self.num_classes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClassConfig':
        """从字典反序列化"""
        return cls(class_names=data.get('class_names'))

    @classmethod
    def binary(cls) -> 'ClassConfig':
        """创建二分类配置（向后兼容）"""
        return cls(['static', 'dynamic'])

    @classmethod
    def default(cls) -> 'ClassConfig':
        """创建默认三分类配置"""
        return cls(DEFAULT_CLASSES)
