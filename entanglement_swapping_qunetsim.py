"""
================================================================================
Mô phỏng Quantum Network và Hoán đổi Cặp vướng bằng QuNetSim
================================================================================
Giao thức Mạng Lượng tử bất đồng bộ (Asynchronous Event-Driven Quantum Network)
Kiến trúc Nút: Alice <--- Kênh Lượng tử/Cổ điển ---> Repeater <--- Kênh Lượng tử/Cổ điển ---> Bob
"""

import sys
import io
import time
from qunetsim.components import Host, Network

# Đảm bảo in Tiếng Việt trên Windows console không bị lỗi Unicode
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer') and not getattr(sys.stdout, '_utf8_wrapped', False):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stdout._utf8_wrapped = True
    if hasattr(sys.stderr, 'buffer') and not getattr(sys.stderr, '_utf8_wrapped', False):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        sys.stderr._utf8_wrapped = True


def alice_protocol(host, repeater_id, bob_id, results_dict):
    """
    Giao thức chạy tại Nút Alice.
    """
    # 1. Khởi tạo cặp EPR với Repeater
    epr_id, ack = host.send_epr(repeater_id, await_ack=True)
    q_alice = host.get_epr(repeater_id, epr_id)
    
    if q_alice is None:
        print("[Alice] LỖI: Không nhận được qubit EPR từ trạm lặp.")
        return

    # 2. Chờ một khoảng thời gian ngắn để trạm lặp thực hiện BSM
    time.sleep(0.5)

    # 3. Đo Qubit của Alice
    m_alice = q_alice.measure()
    results_dict['alice_measurement'] = m_alice

    # 4. Gửi kết quả đo cổ điển cho Bob để đối chiếu
    host.send_classical(bob_id, m_alice)


def repeater_protocol(host, alice_id, bob_id, results_dict):
    """
    Giao thức chạy tại Trạm Lặp (Repeater Station).
    """
    # 1. Nhận qubit EPR từ cặp (Alice - Repeater)
    q_r1 = host.get_epr(alice_id, wait=10)
    if q_r1 is None:
        print("[Repeater] LỖI: Không nhận được Qubit từ Alice.")
        return

    # 2. Khởi tạo cặp EPR với Bob (Repeater - Bob)
    epr_id_2, ack = host.send_epr(bob_id, await_ack=True)
    q_r2 = host.get_epr(bob_id, epr_id_2)
    if q_r2 is None:
        print("[Repeater] LỖI: Không tạo được Qubit EPR với Bob.")
        return

    # 3. Thực hiện phép đo trạng thái Bell (BSM) trên 2 Qubit nội bộ (q_r1, q_r2)
    q_r1.cnot(q_r2)
    q_r1.H()
    m1 = q_r1.measure()
    m2 = q_r2.measure()

    results_dict['bsm_m1'] = m1
    results_dict['bsm_m2'] = m2

    # 4. Gửi thông tin hiệu chỉnh cổ điển (m1, m2) cho Bob
    host.send_classical(bob_id, (m1, m2))


def bob_protocol(host, repeater_id, alice_id, results_dict):
    """
    Giao thức chạy tại Nút Bob.
    """
    # 1. Nhận qubit EPR từ trạm lặp
    q_bob = host.get_epr(repeater_id, wait=10)
    if q_bob is None:
        print("[Bob] LỖI: Không nhận được Qubit từ trạm lặp.")
        return

    # 2. Nhận thông tin hiệu chỉnh cổ điển từ trạm lặp
    msg = host.get_classical(repeater_id, wait=10)
    if not msg:
        print("[Bob] LỖI: Không nhận được tín hiệu cổ điển từ trạm lặp.")
        return

    m1, m2 = msg[0].content

    # 3. Áp dụng các cổng hiệu chỉnh Pauli
    if m2 == 1:
        q_bob.X()
    if m1 == 1:
        q_bob.Z()

    # 4. Đo Qubit của Bob sau khi hiệu chỉnh
    m_bob = q_bob.measure()
    results_dict['bob_measurement'] = m_bob

    # 5. Nhận kết quả đo từ Alice để đối chiếu vướng Lượng tử
    msg_alice = host.get_classical(alice_id, wait=10)
    if msg_alice:
        results_dict['alice_sent_measurement'] = msg_alice[0].content


def run_qunetsim_simulation(num_trials=5):
    """
    Chạy mô phỏng mạng QuNetSim với số lần thử nghiệm (trials).
    """
    print("=======================================================")
    print(f" MÔ PHỎNG QUNETSIM QUANTUM NETWORK (HOÁN ĐỔI CẶP VƯỚNG)")
    print(f" Số lần thử nghiệm: {num_trials}")
    print("=======================================================\n")

    matches = 0

    for trial in range(1, num_trials + 1):
        print(f"--- Thử nghiệm {trial}/{num_trials} ---")
        
        # 1. Khởi tạo Singleton Network
        network = Network.get_instance()
        network.start()

        # 2. Khởi tạo các Hosts (Nút mạng)
        alice = Host('Alice')
        repeater = Host('Repeater')
        bob = Host('Bob')

        # 3. Tạo các kết nối lượng tử & cổ điển giữa các nút
        alice.add_connection('Repeater')
        repeater.add_connection('Alice')
        repeater.add_connection('Bob')
        bob.add_connection('Repeater')
        
        # Kết nối kênh cổ điển trực tiếp giữa Alice và Bob để đối chiếu
        alice.add_connection('Bob')
        bob.add_connection('Alice')

        alice.start()
        repeater.start()
        bob.start()

        network.add_hosts([alice, repeater, bob])

        results = {}

        # 4. Thực thi giao thức trên các luồng nút độc lập
        t1 = alice.run_protocol(alice_protocol, (repeater.host_id, bob.host_id, results))
        t2 = repeater.run_protocol(repeater_protocol, (alice.host_id, bob.host_id, results))
        t3 = bob.run_protocol(bob_protocol, (repeater.host_id, alice.host_id, results))

        t1.join()
        t2.join()
        t3.join()

        # Dừng mạng cho lượt thử nghiệm hiện tại
        network.stop(True)

        m_alice = results.get('alice_measurement')
        m_bob = results.get('bob_measurement')
        m1 = results.get('bsm_m1')
        m2 = results.get('bsm_m2')

        is_matched = (m_alice == m_bob) and (m_alice is not None)
        if is_matched:
            matches += 1

        print(f"  BSM tại Trạm Lặp: (m1={m1}, m2={m2})")
        print(f"  Kết quả đo Alice = {m_alice}, Bob = {m_bob}")
        print(f"  Trạng thái vướng: {'THÀNH CÔNG (Trùng khớp 100%)' if is_matched else 'THẤT BẠI'}\n")

    fidelity = (matches / num_trials) * 100
    print("=======================================================")
    print(f" TỔNG KẾT MÔ PHỎNG QUNETSIM:")
    print(f"  - Số lần thử thành công: {matches}/{num_trials}")
    print(f"  - Độ trung thực (Fidelity): {fidelity:.2f}%")
    print("=======================================================")

if __name__ == '__main__':
    run_qunetsim_simulation(num_trials=5)
