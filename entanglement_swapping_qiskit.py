"""
================================================================================
Mô phỏng Phân phối và Hoán đổi Cặp vướng (Entanglement Swapping) sử dụng Qiskit
================================================================================
Mô hình Trạm lặp Lượng tử (Quantum Repeater Network)
Alice (Node A) <---> Repeater (Node R) <---> Bob (Node B)
"""

import os
import sys
import io

# Đảm bảo in Tiếng Việt trên Windows console không bị lỗi Unicode
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer') and not getattr(sys.stdout, '_utf8_wrapped', False):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stdout._utf8_wrapped = True
    if hasattr(sys.stderr, 'buffer') and not getattr(sys.stderr, '_utf8_wrapped', False):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        sys.stderr._utf8_wrapped = True

import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

def create_single_repeater_circuit(basis='Z'):
    """
    Tạo mạch lượng tử hoán đổi cặp vướng qua 1 trạm lặp (Alice - Repeater - Bob).
    
    Cấu trúc Qubit:
      q[0]: Qubit của Alice (nút A)
      q[1]: Qubit trạm lặp (Cặp EPR 1 với Alice)
      q[2]: Qubit trạm lặp (Cặp EPR 2 với Bob)
      q[3]: Qubit của Bob (nút B)
    """
    qr = QuantumRegister(4, name='q')
    cr_m1 = ClassicalRegister(1, name='m1')    # Kết quả đo q1 tại Repeater (Z correction)
    cr_m2 = ClassicalRegister(1, name='m2')    # Kết quả đo q2 tại Repeater (X correction)
    cr_ab = ClassicalRegister(2, name='meas_AB') # Kết quả đo cuối cùng của Alice & Bob

    qc = QuantumCircuit(qr, cr_m1, cr_m2, cr_ab)

    # --------------------------------------------------------------------------
    # BƯỚC 1: Phân phối các cặp vướng (EPR Distribution)
    # --------------------------------------------------------------------------
    # Cặp EPR 1: Alice (q0) và Repeater (q1) -> |Phi+> = (|00> + |11>) / sqrt(2)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    # Cặp EPR 2: Repeater (q2) và Bob (q3) -> |Phi+> = (|00> + |11>) / sqrt(2)
    qc.h(qr[2])
    qc.cx(qr[2], qr[3])

    qc.barrier()

    # --------------------------------------------------------------------------
    # BƯỚC 2: Đo trạng thái Bell (Bell State Measurement - BSM) tại Trạm lặp
    # --------------------------------------------------------------------------
    # Đo BSM trên 2 qubit (q1, q2) nằm tại trạm lặp
    qc.cx(qr[1], qr[2])
    qc.h(qr[1])
    qc.measure(qr[1], cr_m1[0])
    qc.measure(qr[2], cr_m2[0])

    qc.barrier()

    # --------------------------------------------------------------------------
    # BƯỚC 3: Sửa lỗi Pauli (Classical Feedforward / Pauli Correction) tại Bob
    # --------------------------------------------------------------------------
    # Dựa trên 2 bit cổ điển nhận từ trạm lặp:
    # Nếu m2 = 1 => Bob áp dụng cổng X
    with qc.if_test((cr_m2[0], 1)):
        qc.x(qr[3])

    # Nếu m1 = 1 => Bob áp dụng cổng Z
    with qc.if_test((cr_m1[0], 1)):
        qc.z(qr[3])

    qc.barrier()

    # --------------------------------------------------------------------------
    # BƯỚC 4: Kiểm chứng vướng lượng tử giữa Alice (q0) và Bob (q3)
    # --------------------------------------------------------------------------
    if basis.upper() == 'X':
        # Đổi cơ sở đo sang cơ sở X (Hadamard basis)
        qc.h(qr[0])
        qc.h(qr[3])

    qc.measure(qr[0], cr_ab[0])
    qc.measure(qr[3], cr_ab[1])

    return qc

def simulate_single_repeater(shots=1000, basis='Z', plot_filename='entanglement_swapping_single.png'):
    """
    Chạy mô phỏng 1 trạm lặp và in thống kê chi tiết.
    """
    print(f"\n=======================================================")
    print(f" MÔ PHỎNG HOÁN ĐỔI CẶP VƯỚNG (1 TRẠM LẶP) - Cơ sở đo: {basis}")
    print(f"=======================================================")

    qc = create_single_repeater_circuit(basis=basis)
    
    sim = AerSimulator()
    qc_compiled = qiskit.transpile(qc, sim)
    result = sim.run(qc_compiled, shots=shots).result()
    raw_counts = result.get_counts()

    # Gom nhóm kết quả đo của Alice và Bob
    ab_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    
    # Raw counts key format: "meas_AB m2 m1"
    for key, count in raw_counts.items():
        parts = key.split()
        ab_bits = parts[0] # bit của Alice (phải) & Bob (trái)
        ab_counts[ab_bits] += count

    total_valid = ab_counts['00'] + ab_counts['11']
    fidelity = total_valid / shots

    print("\n[MẠCH LƯỢNG TỬ]:")
    print(qc.draw(output='text'))

    print("\n[KẾT QUẢ ĐO CỦA ALICE & BOB]:")
    print(f"  - Trạng thái |00>: {ab_counts['00']} lần ({ab_counts['00']/shots*100:.2f}%)")
    print(f"  - Trạng thái |11>: {ab_counts['11']} lần ({ab_counts['11']/shots*100:.2f}%)")
    print(f"  - Trạng thái |01>: {ab_counts['01']} lần ({ab_counts['01']/shots*100:.2f}%) [Lỗi / Không vướng]")
    print(f"  - Trạng thái |10>: {ab_counts['10']} lần ({ab_counts['10']/shots*100:.2f}%) [Lỗi / Không vướng]")
    print(f"\n=> ĐỘ TRUNG THỰC (FIDELITY |Phi+>): {fidelity * 100:.2f}%")
    
    if fidelity > 0.99:
        print("=> XÁC NHẬN: Alice và Bob đã đạt vướng lượng tử hoàn hảo mặc dù chưa từng tương tác trực tiếp!")

    # Vẽ và lưu đồ thị histogram
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(ab_counts.keys(), ab_counts.values(), color=['#2b5c8f', '#d9534f', '#d9534f', '#2b5c8f'])
        ax.set_title(f"Hoán đổi cặp vướng (1 Trạm Lặp) - Cơ sở đo {basis}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Trạng thái đo (Bob Alice)", fontsize=12)
        ax.set_ylabel("Số lần xuất hiện (Shots)", fontsize=12)
        ax.set_ylim(0, shots)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval} ({yval/shots*100:.1f}%)", ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        print(f"-> Đã lưu biểu đồ kết quả tại: {os.path.abspath(plot_filename)}")
    except Exception as e:
        print(f"Không thể tạo file ảnh biểu đồ: {e}")

    return fidelity, ab_counts

def create_multi_repeater_circuit(num_repeaters=2):
    """
    Tạo mạch hoán đổi cặp vướng qua chuỗi N trạm lặp.
    Ví dụ N=2: Alice -> R1 -> R2 -> Bob (6 qubits).
    """
    total_qubits = 2 * (num_repeaters + 1)
    qr = QuantumRegister(total_qubits, name='q')
    
    # Tạo các thanh ghi đo cho từng trạm lặp
    repeaters_crs = []
    for r in range(num_repeaters):
        cr1 = ClassicalRegister(1, name=f'r{r+1}_m1')
        cr2 = ClassicalRegister(1, name=f'r{r+1}_m2')
        repeaters_crs.append((cr1, cr2))

    cr_ab = ClassicalRegister(2, name='meas_AB')

    all_crs = []
    for cr1, cr2 in repeaters_crs:
        all_crs.extend([cr1, cr2])
    all_crs.append(cr_ab)

    qc = QuantumCircuit(qr, *all_crs)

    # 1. Tạo các cặp vướng giữa các nút kề nhau
    # Cặp 0: (q0, q1) -> Alice & R1
    # Cặp 1: (q2, q3) -> R1 & R2
    # Cặp 2: (q4, q5) -> R2 & Bob
    for i in range(num_repeaters + 1):
        q_left = 2 * i
        q_right = 2 * i + 1
        qc.h(qr[q_left])
        qc.cx(qr[q_left], qr[q_right])

    qc.barrier()

    # 2. BSM tại các trạm lặp
    for r in range(num_repeaters):
        q_r_left = 2 * r + 1   # Qubit thuộc cặp bên trái tại trạm r+1
        q_r_right = 2 * (r + 1) # Qubit thuộc cặp bên phải tại trạm r+1
        
        cr1, cr2 = repeaters_crs[r]
        qc.cx(qr[q_r_left], qr[q_r_right])
        qc.h(qr[q_r_left])
        qc.measure(qr[q_r_left], cr1[0])
        qc.measure(qr[q_r_right], cr2[0])

    qc.barrier()

    # 3. Phối hợp hiệu chỉnh Pauli tại Bob (Qubit cuối cùng = total_qubits - 1)
    bob_qubit = total_qubits - 1

    # Hiệu chỉnh X: Áp dụng X nếu tổng số bit m2 là số lẻ
    for r in range(num_repeaters):
        cr1, cr2 = repeaters_crs[r]
        with qc.if_test((cr2[0], 1)):
            qc.x(qr[bob_qubit])

    # Hiệu chỉnh Z: Áp dụng Z nếu tổng số bit m1 là số lẻ
    for r in range(num_repeaters):
        cr1, cr2 = repeaters_crs[r]
        with qc.if_test((cr1[0], 1)):
            qc.z(qr[bob_qubit])

    qc.barrier()

    # 4. Đo Alice (q0) và Bob (qubit cuối)
    qc.measure(qr[0], cr_ab[0])
    qc.measure(qr[bob_qubit], cr_ab[1])

    return qc

def simulate_multi_repeater(num_repeaters=2, shots=1000, plot_filename='entanglement_swapping_multi.png'):
    """
    Chạy mô phỏng chuỗi nhiều trạm lặp.
    """
    print(f"\n=======================================================")
    print(f" MÔ PHỎNG CHUỖI QUANTUM REPEATER ({num_repeaters} TRẠM LẶP)")
    print(f" Sơ đồ: Alice <-> R1 <-> ... <-> R{num_repeaters} <-> Bob")
    print(f"=======================================================")

    qc = create_multi_repeater_circuit(num_repeaters=num_repeaters)
    
    sim = AerSimulator()
    qc_compiled = qiskit.transpile(qc, sim)
    result = sim.run(qc_compiled, shots=shots).result()
    raw_counts = result.get_counts()

    ab_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    for key, count in raw_counts.items():
        parts = key.split()
        ab_bits = parts[0]
        ab_counts[ab_bits] += count

    total_valid = ab_counts['00'] + ab_counts['11']
    fidelity = total_valid / shots

    print(f"\n[KẾT QUẢ ĐO CỦA ALICE & BOB QUA {num_repeaters} TRẠM LẶP]:")
    print(f"  - Trạng thái |00>: {ab_counts['00']} lần ({ab_counts['00']/shots*100:.2f}%)")
    print(f"  - Trạng thái |11>: {ab_counts['11']} lần ({ab_counts['11']/shots*100:.2f}%)")
    print(f"  - Trạng thái |01>: {ab_counts['01']} lần ({ab_counts['01']/shots*100:.2f}%)")
    print(f"  - Trạng thái |10>: {ab_counts['10']} lần ({ab_counts['10']/shots*100:.2f}%)")
    print(f"\n=> ĐỘ TRUNG THỰC (FIDELITY |Phi+>): {fidelity * 100:.2f}%")

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(ab_counts.keys(), ab_counts.values(), color=['#2b5c8f', '#d9534f', '#d9534f', '#2b5c8f'])
        ax.set_title(f"Hoán đổi cặp vướng qua {num_repeaters} Trạm Lặp", fontsize=14, fontweight='bold')
        ax.set_xlabel("Trạng thái đo (Bob Alice)", fontsize=12)
        ax.set_ylabel("Số lần xuất hiện (Shots)", fontsize=12)
        ax.set_ylim(0, shots)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval} ({yval/shots*100:.1f}%)", ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        print(f"-> Đã lưu biểu đồ kết quả tại: {os.path.abspath(plot_filename)}")
    except Exception as e:
        print(f"Không thể tạo file ảnh biểu đồ: {e}")

    return fidelity, ab_counts

if __name__ == '__main__':
    # Chạy mô phỏng đơn
    simulate_single_repeater(shots=1000, basis='Z')
    simulate_single_repeater(shots=1000, basis='X')
    
    # Chạy mô phỏng chuỗi 2 trạm lặp
    simulate_multi_repeater(num_repeaters=2, shots=1000)
