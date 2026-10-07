from qunetsim.components import Host, Network
from qunetsim.objects import Qubit

def main():
    # 1. Khởi tạo mạng Singleton
    network = Network.get_instance()
    network.start()

    # 2. Khởi tạo các Host
    alice = Host('Alice')
    bob = Host('Bob')

    # 3. Tạo kết nối giữa hai nút
    alice.add_connection('Bob')
    bob.add_connection('Alice')

    alice.start()
    bob.start()

    network.add_hosts([alice, bob])

    # 4. Tạo qubit và gửi từ Alice sang Bob
    q = Qubit(alice)
    q.X()  # Lật qubit từ |0> sang |1>
    alice.send_qubit('Bob', q)

    # 5. Bob nhận và đo qubit
    received_q = bob.get_qubit('Alice', wait=5)
    if received_q:
        print("Bob nhan duoc qubit voi gia tri do:", received_q.measure())
    else:
        print("Khong nhan duoc qubit!")

    # 6. Dừng mạng
    network.stop(True)

if __name__ == '__main__':
    main()