import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    # Setup Figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Helper to draw boxes
    def draw_box(x, y, w, h, text, color='#E0E0E0', edge='black', style='solid', label_color='black'):
        # Shadow
        shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1", 
                                       ec="none", fc='#D3D3D3', alpha=0.5, mutation_scale=20)
        ax.add_patch(shadow)
        # Main Box
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     ec=edge, fc=color, linestyle=style, linewidth=1.5, mutation_scale=20)
        ax.add_patch(box)
        # Text
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', color=label_color, wrap=True)
        return box

    # Helper for arrows
    def draw_arrow(x1, y1, x2, y2, color='gray'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

    # --- 1. HV POWER PATH (Top Row) ---
    # Input
    ax.text(0.5, 6, "270V DC\nInput", ha='center', va='center', fontsize=10, fontweight='bold')
    draw_arrow(0.8, 6, 1.2, 6, 'black')

    # BLUE: HV Input Filter
    draw_box(1.2, 5.5, 1.5, 1.0, "HV Input Filter\n(CMC + TVS + Caps)", color='#ADD8E6', edge='#00008B') # Blue
    
    draw_arrow(2.8, 6, 3.2, 6)
    
    # Grey: DC Link & Inverter
    draw_box(3.2, 5.5, 1.2, 1.0, "DC Link\nCapacitor", color='#F0F0F0', edge='gray', style='dashed', label_color='gray')
    draw_arrow(4.5, 6, 5.0, 6, 'gray')
    draw_box(5.0, 5.5, 1.5, 1.0, "IGBT Inverter", color='#F0F0F0', edge='gray', style='dashed', label_color='gray')
    draw_arrow(6.6, 6, 7.5, 6, 'gray')
    
    # Motor
    circle = patches.Circle((8.0, 6.0), 0.5, ec='gray', fc='#F0F0F0', ls='dashed')
    ax.add_patch(circle)
    ax.text(8.0, 6.0, "Motor", ha='center', va='center', color='gray')

    # --- 2. LV POWER PATH (Middle Row) ---
    # Down from DC Link to Buck
    draw_arrow(3.8, 5.5, 3.8, 4.5, 'gray')
    
    # Grey: DC/DC Buck
    draw_box(3.2, 3.5, 1.2, 1.0, "Isolated\nDC/DC\n(270V->24V)", color='#F0F0F0', edge='gray', style='dashed', label_color='gray')
    draw_arrow(4.5, 4, 5.0, 4)

    # ORANGE: LV Decoupling
    draw_box(5.0, 3.5, 1.5, 1.0, "LV Interstage\nFilter\n(Ripple Rejection)", color='#FFDAB9', edge='#FF8C00') # Orange
    draw_arrow(6.6, 4, 7.2, 4, 'gray')

    # Grey: Gate Drivers
    draw_box(7.2, 3.5, 1.5, 1.0, "Gate Drivers", color='#F0F0F0', edge='gray', style='dashed', label_color='gray')
    # Arrow up to Inverter
    draw_arrow(7.95, 4.6, 6.0, 5.5, 'gray')

    # --- 3. SIGNAL & LOGIC PATH (Bottom Row) ---
    # Power down to MCU
    draw_arrow(5.75, 3.5, 5.75, 2.5, 'gray')

    # Grey: MCU
    draw_box(5.0, 1.5, 1.5, 1.0, "MCU / Logic", color='#F0F0F0', edge='gray', style='dashed', label_color='gray')

    # --- SIGNAL FILTERS ---
    
    # Encoder Input
    ax.text(8.5, 0.5, "Encoder\n(Motor)", ha='center', va='center', color='gray', fontsize=9)
    draw_arrow(8.5, 0.8, 8.5, 1.5, 'gray') # Up from sensor
    
    # GREEN: Encoder Filter
    draw_box(7.5, 1.5, 1.2, 1.0, "Signal Filter\n(RC Network)", color='#C1FFC1', edge='#006400') # Green
    draw_arrow(7.5, 2.0, 6.6, 2.0, 'gray') # Into MCU

    # GREEN: Analog Output Filter
    draw_box(2.5, 1.5, 1.5, 1.0, "Output Filter\n(Ferrites)", color='#C1FFC1', edge='#006400') # Green
    draw_arrow(5.0, 2.0, 4.1, 2.0, 'gray') # Out from MCU
    
    # Output Connector
    draw_arrow(2.5, 2.0, 1.5, 2.0, 'black')
    ax.text(1.0, 2.0, "Analog\nOutput\n(4-20mA)", ha='center', va='center', fontsize=10, fontweight='bold')

    # --- LEGEND ---
    ax.text(1.95, 0.5, "BLUE: HV Surge/EMI", color='#00008B', fontweight='bold')
    ax.text(5.0, 0.5, "ORANGE: LV Stability", color='#D2691E', fontweight='bold')
    ax.text(8.0, 0.5, "GREEN: Signal Integrity", color='#006400', fontweight='bold')
    
    plt.title("Figure 3.7.1: Simplified Filtering Architecture", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("filtering_architecture.png", dpi=300)
    plt.show()

create_architecture_diagram()