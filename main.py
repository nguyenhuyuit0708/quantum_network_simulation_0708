"""
================================================================================
CHƯƠNG TRÌNH MÔ PHỎNG QUANTUM NETWORK & ENTANGLEMENT SWAPPING
================================================================================
"""

import sys
import io

if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer') and not getattr(sys.stdout, '_utf8_wrapped', False):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stdout._utf8_wrapped = True
    if hasattr(sys.stderr, 'buffer') and not getattr(sys.stderr, '_utf8_wrapped', False):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        sys.stderr._utf8_wrapped = True

from entanglement_swapping_qiskit import simulate_single_repeater, simulate_multi_repeater
from entanglement_swapping_qunetsim import run_qunetsim_simulation
from quantum_routing import run_routing_demonstration, run_fidelity_decay_evaluation

def main():
    print("=================================================================")
    print("      MÔ PHỎNG PHÂN PHỐI VÀ HOÁN ĐỔI CẶP VƯỚNG (QUANTUM NETWORK)")
    print("=================================================================")
    print("  1. Mô phỏng Qiskit: 1 Trạm lặp (Cơ sở Z & Cơ sở X)")
    print("  2. Mô phỏng Qiskit: Chuỗi 2 Trạm lặp (Alice -> R1 -> R2 -> Bob)")
    print("  3. Mô phỏng QuNetSim: Giao thức mạng bất đồng bộ (5 thử nghiệm)")
    print("  4. Định tuyến mạng lượng tử: Dijkstra vs. Tối ưu độ trung thực (Fidelity-Aware)")
    print("  5. Đánh giá mức độ suy giảm trạng thái lượng tử qua từng chặng (Fidelity Decay)")
    print("  6. Chạy toàn bộ tất cả mô phỏng & phân tích (1 -> 5)")
    print("  0. Thoát")
    print("=================================================================")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Nhập lựa chọn của bạn (0-6) [Mặc định: 5]: ").strip()
        if not choice:
            choice = '5'

    if choice == '1':
        simulate_single_repeater(shots=1000, basis='Z')
        simulate_single_repeater(shots=1000, basis='X')
    elif choice == '2':
        simulate_multi_repeater(num_repeaters=2, shots=1000)
    elif choice == '3':
        run_qunetsim_simulation(num_trials=5)
    elif choice == '4':
        run_routing_demonstration()
    elif choice == '5':
        run_fidelity_decay_evaluation()
    elif choice == '6':
        print("\n>>> [1/5] ĐANG CHẠY MÔ PHỎNG QISKIT (1 TRẠM LẶP)...")
        simulate_single_repeater(shots=1000, basis='Z')
        simulate_single_repeater(shots=1000, basis='X')

        print("\n>>> [2/5] ĐANG CHẠY MÔ PHỎNG QISKIT (2 TRẠM LẶP)...")
        simulate_multi_repeater(num_repeaters=2, shots=1000)

        print("\n>>> [3/5] ĐANG CHẠY MÔ PHỎNG QUNETSIM (QUANTUM NETWORK)...")
        run_qunetsim_simulation(num_trials=5)

        print("\n>>> [4/5] ĐANG CHẠY ĐỊNH TUYẾN MẠNG LƯỢNG TỬ (DIJKSTRA & FIDELITY-AWARE)...")
        run_routing_demonstration()

        print("\n>>> [5/5] ĐANG ĐÁNH GIÁ MỨC ĐỘ SUY GIẢM TRẠNG THÁI LƯỢNG TỬ (FIDELITY DECAY)...")
        run_fidelity_decay_evaluation()
    else:
        print("Đã thoát chương trình.")

if __name__ == '__main__':
    main()
