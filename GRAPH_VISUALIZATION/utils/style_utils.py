def setup_matplotlib_style():
    """设置matplotlib的样式"""
    import matplotlib.pyplot as plt
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    return {
        'node': {
            'default_size': 500,
            'highlight_size': 700,
            'default_color': 'lightblue',
            'highlight_color': '#ff7f0e',
        },
        'edge': {
            'default_width': 1,
            'highlight_width': 2,
            'default_color': 'gray',
            'highlight_color': 'red',
        },
        'figure': {
            'background': '#f0f0f0',
            'face_color': 'white',
            'edge_color': 'gray',
            'edge_width': 1,
        }
    }