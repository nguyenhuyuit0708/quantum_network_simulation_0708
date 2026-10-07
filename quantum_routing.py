r"""
================================================================================
THUẬT TOÁN ĐỊNH TUYẾN MẠNG LƯỢNG TỬ (QUANTUM NETWORK ROUTING)
================================================================================
1. Thuật toán Dijkstra tìm đường ngắn nhất:
   - Theo khoảng cách vật lý (Physical Distance - km)
   - Theo số chặng (Hop Count)
2. Thuật toán Dijkstra tối ưu độ trung thực (Fidelity-Aware Routing):
   - Chuyển đổi độ trung thực liên kết (Link Fidelity F_e) thành trọng số cộng tính:
     W_fidelity(e) = -ln(p_e), với p_e = (4*F_e - 1) / 3 (Tham số Werner/Depolarizing).
   - Tối thiểu hóa tổng trọng số W tương đương với tối đa hóa độ trung thực cuối-cuối:
     F_e2e = (1 + 3 * \prod p_e) / 4
3. Trực quan hóa cấu trúc liên kết mạng và tuyến đường tối ưu (NetworkX + Matplotlib).
4. Mô phỏng hoán đổi cặp vướng (Entanglement Swapping) trên tuyến đường bằng Qiskit Aer
   với mô hình nhiễu Depolarizing chính xác cho từng liên kết.
================================================================================
"""

import os
import sys
import io
import math
import heapq
import matplotlib.pyplot as plt
import networkx as nx

if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer') and not getattr(sys.stdout, '_utf8_wrapped', False):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stdout._utf8_wrapped = True
    if hasattr(sys.stderr, 'buffer') and not getattr(sys.stderr, '_utf8_wrapped', False):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        sys.stderr._utf8_wrapped = True

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


class QuantumNetworkGraph:
    """
    Biểu diễn đồ thị mạng lượng tử với các nút (Hosts / Quantum Repeaters)
    và các liên kết lượng tử (Quantum Links) có thông số vật lý:
    - distance: Khoảng cách vật lý (km)
    - fidelity: Độ trung thực cặp EPR ban đầu F_e in (0.25, 1.0]
    - werner_p: Tham số depolarizing Werner p_e = (4*F_e - 1) / 3
    - weight_fidelity: Trọng số -ln(p_e) dùng cho thuật toán Dijkstra
    """

    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id, node_type="Repeater", pos=None):
        """Thêm một nút vào mạng lượng tử."""
        self.graph.add_node(node_id, node_type=node_type, pos=pos)

    def add_link(self, u, v, distance_km, fidelity):
        """
        Thêm một kênh lượng tử giữa hai nút u và v.
        
        Mô hình toán học Werner:
            Trạng thái Werner: rho = p |Phi+><Phi+| + (1-p)/4 * I
            Độ trung thực: F = (1 + 3*p) / 4  =>  p = (4*F - 1) / 3
            Trọng số cộng tính Dijkstra: W_fidelity = -ln(p)
        """
        if not (0.25 < fidelity <= 1.0):
            raise ValueError(f"Fidelity phải nằm trong khoảng (0.25, 1.0], nhận: {fidelity}")

        p = (4.0 * fidelity - 1.0) / 3.0
        p = max(p, 1e-6)
        w_fidelity = -math.log(p)

        self.graph.add_edge(
            u, v,
            distance=float(distance_km),
            fidelity=float(fidelity),
            werner_p=float(p),
            weight_fidelity=float(w_fidelity)
        )

    @classmethod
    def create_default_topology(cls):
        """
        Khởi tạo topo mạng lượng tử đa nút mở rộng (8 nút):
        - Alice (Nút nguồn)
        - Bob (Nút đích)
        - 6 Trạm lặp trung gian: R1, R2, R3, R4, R5, R6
        
        Kịch bản thực tế:
        - Tuyến ngắn nhất (Distance Dijkstra): Alice -> R1 -> Bob (2 chặng, 80 km)
          Đi qua kênh đô thị cáp quang có độ suy hao và nhiễu lớn (Fidelity ~ 0.82 - 0.83).
        - Tuyến tối ưu độ trung thực (Fidelity-Aware Dijkstra): Alice -> R2 -> R3 -> Bob (3 chặng, 105 km)
          Đi qua tuyến cáp quang chất lượng cao / trạm lặp có bộ nhớ lượng tử tốt (Fidelity ~ 0.97 - 0.98).
        - Các liên kết phụ tạo thành cấu trúc Mesh Network phong phú.
        """
        net = cls()

        positions = {
            'Alice': (0.0, 2.0),
            'R1': (3.0, 3.2),      # Tuyến ngắn nhưng nhiều nhiễu
            'R2': (2.0, 0.8),      # Tuyến cáp chất lượng cao
            'R3': (4.5, 0.8),
            'R4': (1.5, 3.5),
            'R5': (3.2, -0.4),
            'R6': (5.0, 3.2),
            'Bob': (6.5, 2.0)
        }

        # Đăng ký các nút
        net.add_node('Alice', node_type='EndNode', pos=positions['Alice'])
        net.add_node('Bob', node_type='EndNode', pos=positions['Bob'])
        for r_id in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
            net.add_node(r_id, node_type='Repeater', pos=positions[r_id])

        # Tuyến 1: Ngắn nhất theo km nhưng fidelity thấp
        net.add_link('Alice', 'R1', distance_km=40.0, fidelity=0.82)
        net.add_link('R1', 'Bob', distance_km=40.0, fidelity=0.83)

        # Tuyến 2: 3 chặng, 105 km nhưng fidelity rất cao
        net.add_link('Alice', 'R2', distance_km=35.0, fidelity=0.98)
        net.add_link('R2', 'R3', distance_km=35.0, fidelity=0.97)
        net.add_link('R3', 'Bob', distance_km=35.0, fidelity=0.98)

        # Các liên kết lưới phụ
        net.add_link('Alice', 'R4', distance_km=30.0, fidelity=0.88)
        net.add_link('R4', 'R1', distance_km=25.0, fidelity=0.85)
        net.add_link('R1', 'R6', distance_km=30.0, fidelity=0.86)
        net.add_link('R6', 'Bob', distance_km=25.0, fidelity=0.89)

        net.add_link('R2', 'R5', distance_km=28.0, fidelity=0.93)
        net.add_link('R5', 'R3', distance_km=30.0, fidelity=0.94)
        net.add_link('R1', 'R3', distance_km=45.0, fidelity=0.79)

        return net

    def dijkstra(self, source, target, weight_key='distance'):
        """
        Thuật toán tìm đường ngắn nhất Dijkstra:
        - weight_key='distance': Tối thiểu tổng khoảng cách (km)
        - weight_key='hops': Tối thiểu số chặng (Hops)
        - weight_key='weight_fidelity': Tối đa độ trung thực cuối-cuối (-ln p)
        """
        if source not in self.graph:
            raise ValueError(f"Nút nguồn '{source}' không tồn tại trong mạng.")
        if target not in self.graph:
            raise ValueError(f"Nút đích '{target}' không tồn tại trong mạng.")

        distances = {node: float('inf') for node in self.graph.nodes}
        previous = {node: None for node in self.graph.nodes}
        distances[source] = 0.0

        pq = [(0.0, source)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node == target:
                break

            if current_dist > distances[current_node]:
                continue

            for neighbor in self.graph.neighbors(current_node):
                edge_data = self.graph[current_node][neighbor]
                if weight_key == 'hops':
                    edge_cost = 1.0
                else:
                    edge_cost = edge_data.get(weight_key, 1.0)

                new_dist = current_dist + edge_cost
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        if distances[target] == float('inf'):
            return None, float('inf')

        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = previous[curr]
        path.reverse()

        return path, distances[target]

    def calculate_path_metrics(self, path):
        r"""
        Tính toán toàn bộ các chỉ số vật lý lượng tử cho tuyến đường:
        - Hops (Số chặng)
        - Total Distance (km)
        - Link Fidelities
        - End-to-End Werner Parameter: p_e2e = \prod p_i
        - End-to-End Fidelity: F_e2e = (1 + 3 * p_e2e) / 4
        """
        if not path or len(path) < 2:
            return None

        hops = len(path) - 1
        total_distance = 0.0
        link_fidelities = []
        product_werner_p = 1.0

        for i in range(hops):
            u, v = path[i], path[i + 1]
            edge = self.graph[u][v]
            total_distance += edge['distance']
            f = edge['fidelity']
            p = edge['werner_p']
            link_fidelities.append(f)
            product_werner_p *= p

        end_to_end_fidelity = (1.0 + 3.0 * product_werner_p) / 4.0

        return {
            'path': path,
            'hops': hops,
            'total_distance_km': total_distance,
            'link_fidelities': link_fidelities,
            'product_werner_p': product_werner_p,
            'end_to_end_fidelity': end_to_end_fidelity
        }

    def find_shortest_distance_route(self, source='Alice', target='Bob'):
        """Định tuyến Dijkstra theo khoảng cách vật lý (km)."""
        path, _ = self.dijkstra(source, target, weight_key='distance')
        return self.calculate_path_metrics(path)

    def find_shortest_hop_route(self, source='Alice', target='Bob'):
        """Định tuyến Dijkstra theo số bước nhảy (Hops)."""
        path, _ = self.dijkstra(source, target, weight_key='hops')
        return self.calculate_path_metrics(path)

    def find_fidelity_aware_route(self, source='Alice', target='Bob'):
        """Định tuyến Dijkstra tối ưu độ trung thực (Fidelity-Aware)."""
        path, _ = self.dijkstra(source, target, weight_key='weight_fidelity')
        return self.calculate_path_metrics(path)

    def visualize_routes(self, shortest_metrics, fidelity_metrics, save_path='quantum_routing_topology.png'):
        """
        Vẽ đồ thị mạng lượng tử và lưu file ảnh sắc nét, làm nổi bật 2 tuyến đường so sánh.
        """
        plt.figure(figsize=(12, 7.2))
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos:
            pos = nx.spring_layout(self.graph, seed=42)

        # Màu sắc cho các loại nút
        node_colors = []
        for node in self.graph.nodes:
            if node in ['Alice', 'Bob']:
                node_colors.append('#2ca02c')  # Nút nguồn/đích: xanh lá cây
            else:
                node_colors.append('#1f77b4')  # Trạm lặp: xanh lam

        # 1. Vẽ các nút
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=1300,
            node_color=node_colors,
            edgecolors='#222222',
            linewidths=2.2
        )
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=11,
            font_weight='bold',
            font_color='white'
        )

        # 2. Vẽ tất cả các liên kết nền
        nx.draw_networkx_edges(
            self.graph, pos,
            width=1.6,
            edge_color='#b0b0b0',
            style='dashed'
        )

        # 3. Làm nổi bật Tuyến ngắn nhất (Màu đỏ cam đậm)
        sp_path = shortest_metrics['path']
        sp_edges = [(sp_path[i], sp_path[i + 1]) for i in range(len(sp_path) - 1)]
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=sp_edges,
            width=4.5,
            edge_color='#d62728',
            alpha=0.85,
            label='Đường ngắn nhất (Shortest Distance Dijkstra)'
        )

        # 4. Làm nổi bật Tuyến tối ưu Fidelity (Màu xanh navy đậm)
        fa_path = fidelity_metrics['path']
        fa_edges = [(fa_path[i], fa_path[i + 1]) for i in range(len(fa_path) - 1)]
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=fa_edges,
            width=3.2,
            edge_color='#1a365d',
            style='solid',
            alpha=0.95,
            label='Đường tối ưu Fidelity (Fidelity-Aware Routing)'
        )

        # 5. Nhãn thông tin trên các cạnh (Khoảng cách & Độ trung thực F)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            edge_labels[(u, v)] = f"{data['distance']:.0f}km\nF={data['fidelity']:.2f}"
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=8.5,
            font_color='#222222'
        )

        # Tiêu đề biểu đồ
        plt.title(
            "ĐỊNH TUYẾN MẠNG LƯỢNG TỬ (QUANTUM NETWORK ROUTING)\n"
            "So sánh: Đường ngắn nhất (Dijkstra) vs Tuyến tối ưu độ trung thực (Fidelity-Aware)",
            fontsize=13,
            fontweight='bold',
            pad=18
        )

        # Hộp thông tin so sánh chi tiết
        info_text = (
            f" [●] Tuyến ngắn nhất (Dijkstra):\n"
            f"     • Lộ trình: {' -> '.join(sp_path)}\n"
            f"     • Khoảng cách: {shortest_metrics['total_distance_km']:.1f} km ({shortest_metrics['hops']} chặng)\n"
            f"     • Fidelity lý thuyết: {shortest_metrics['end_to_end_fidelity'] * 100:.2f}%\n\n"
            f" [●] Tuyến tối ưu Fidelity (Fidelity-Aware):\n"
            f"     • Lộ trình: {' -> '.join(fa_path)}\n"
            f"     • Khoảng cách: {fidelity_metrics['total_distance_km']:.1f} km ({fidelity_metrics['hops']} chặng)\n"
            f"     • Fidelity lý thuyết: {fidelity_metrics['end_to_end_fidelity'] * 100:.2f}%"
        )
        plt.gca().text(
            0.02, 0.05, info_text,
            transform=plt.gca().transAxes,
            fontsize=9.5,
            family='sans-serif',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#f8f9fa', edgecolor='#999999', alpha=0.95)
        )

        plt.legend(loc='upper right', frameon=True, fontsize=10)
        plt.axis('off')
        plt.margins(0.12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"-> Đã lưu sơ đồ trực quan định tuyến tại: {os.path.abspath(save_path)}")


def build_swapping_circuit_for_route(path, basis='Z'):
    """
    Tạo mạch Qiskit tổng quát cho chuỗi hoán đổi cặp vướng dọc theo tuyến đường bất kỳ.
    - path: danh sách các nút, ví dụ ['Alice', 'R2', 'R3', 'Bob'] (hops = 3)
    - basis: 'Z' (cơ sở chuẩn) hoặc 'X' (cơ sở Hadamard) để kiểm chứng vướng toàn diện.
    """
    hops = len(path) - 1
    total_qubits = 2 * hops
    num_repeaters = hops - 1

    qr = QuantumRegister(total_qubits, name='q')

    repeaters_crs = []
    for r in range(num_repeaters):
        cr1 = ClassicalRegister(1, name=f'r{r + 1}_m1')
        cr2 = ClassicalRegister(1, name=f'r{r + 1}_m2')
        repeaters_crs.append((cr1, cr2))

    cr_ab = ClassicalRegister(2, name='meas_AB')

    all_crs = []
    for cr1, cr2 in repeaters_crs:
        all_crs.extend([cr1, cr2])
    all_crs.append(cr_ab)

    qc = QuantumCircuit(qr, *all_crs)

    # 1. Chuẩn bị các cặp EPR ban đầu cho từng liên kết kề nhau
    for i in range(hops):
        q_left = 2 * i
        q_right = 2 * i + 1
        qc.h(qr[q_left])
        qc.cx(qr[q_left], qr[q_right])

    qc.barrier()

    # 2. Thực hiện phép đo BSM tại mỗi trạm lặp trung gian
    for r in range(num_repeaters):
        q_r_left = 2 * r + 1
        q_r_right = 2 * (r + 1)
        cr1, cr2 = repeaters_crs[r]

        qc.cx(qr[q_r_left], qr[q_r_right])
        qc.h(qr[q_r_left])
        qc.measure(qr[q_r_left], cr1[0])
        qc.measure(qr[q_r_right], cr2[0])

    qc.barrier()

    # 3. Phối hợp hiệu chỉnh Pauli tại nút đích Bob (Qubit cuối cùng)
    bob_qubit = total_qubits - 1
    for r in range(num_repeaters):
        cr1, cr2 = repeaters_crs[r]
        with qc.if_test((cr2[0], 1)):
            qc.x(qr[bob_qubit])
        with qc.if_test((cr1[0], 1)):
            qc.z(qr[bob_qubit])

    qc.barrier()

    # 4. Đo qubit của Alice và Bob
    if basis.upper() == 'X':
        qc.h(qr[0])
        qc.h(qr[bob_qubit])

    qc.measure(qr[0], cr_ab[0])
    qc.measure(qr[bob_qubit], cr_ab[1])

    return qc


def simulate_route_fidelity_qiskit(route_metrics, shots=2000):
    """
    Mô phỏng kênh lượng tử có nhiễu (Depolarizing Noise Model) tương ứng
    với độ trung thực của từng liên kết trên tuyến đường.
    
    Quy tắc vật lý:
    - Với cặp EPR có độ trung thực F_i, ta áp dụng kênh depolarizing 2-qubit
      ngay sau cổng CX tạo cặp với tham số lambda_i = 4/3 * (1 - F_i).
    - Sau đó thực hiện BSM và Pauli correction.
    - Đo trên cả 2 cơ sở Z và X để tính toán độ trung thực thực nghiệm:
      P_same = (P_same_Z + P_same_X) / 2
      F_measured = (3 * P_same - 1) / 2
    """
    path = route_metrics['path']
    link_fidelities = route_metrics['link_fidelities']
    hops = route_metrics['hops']

    # Xây dựng mô hình nhiễu
    noise_model = NoiseModel()
    for i in range(hops):
        f = link_fidelities[i]
        # Tham số 2-qubit depolarizing: lambda = 4/3 * (1 - F)
        depol_lambda = (4.0 / 3.0) * (1.0 - f)
        depol_lambda = max(0.0, min(0.74, depol_lambda))
        if depol_lambda > 1e-5:
            err = depolarizing_error(depol_lambda, 2)
            noise_model.add_quantum_error(err, ['cx'], [2 * i, 2 * i + 1])

    sim = AerSimulator(noise_model=noise_model)

    # 1. Chạy trên cơ sở Z
    qc_z = build_swapping_circuit_for_route(path, basis='Z')
    res_z = sim.run(qiskit.transpile(qc_z, sim), shots=shots).result()
    counts_z = res_z.get_counts()

    ab_z = {'00': 0, '01': 0, '10': 0, '11': 0}
    for k, v in counts_z.items():
        ab_bits = k.split()[0]
        ab_z[ab_bits] += v
    same_z = (ab_z['00'] + ab_z['11']) / shots

    # 2. Chạy trên cơ sở X
    qc_x = build_swapping_circuit_for_route(path, basis='X')
    res_x = sim.run(qiskit.transpile(qc_x, sim), shots=shots).result()
    counts_x = res_x.get_counts()

    ab_x = {'00': 0, '01': 0, '10': 0, '11': 0}
    for k, v in counts_x.items():
        ab_bits = k.split()[0]
        ab_x[ab_bits] += v
    same_x = (ab_x['00'] + ab_x['11']) / shots

    # Tính độ trung thực thực nghiệm hoàn chỉnh của trạng thái Bell |Phi+>
    p_same_avg = (same_z + same_x) / 2.0
    measured_fidelity = (3.0 * p_same_avg - 1.0) / 2.0
    measured_fidelity = max(0.0, min(1.0, measured_fidelity))

    return {
        'measured_fidelity': measured_fidelity,
        'p_same_avg': p_same_avg,
        'same_z': same_z,
        'same_x': same_x,
        'counts_z': ab_z,
        'counts_x': ab_x
    }


def evaluate_stepwise_fidelity_decay(route_metrics, shots=2000):
    r"""
    Đánh giá mức độ suy giảm trạng thái lượng tử (Fidelity Decay) qua từng chặng truyền dẫn:
    - Tính toán lý thuyết tích lũy: F^(m) = (1 + 3 * \prod_{i=1}^m p_i) / 4
    - Mô phỏng thực nghiệm từng bước trên Qiskit Aer để kiểm chứng:
      * Chặng 1: Alice - R_1 (chưa BSM)
      * Chặng 2: Alice - R_2 (sau 1 BSM tại R_1)
      * Chặng 3: Alice - R_3 (sau 2 BSMs) ...
    """
    path = route_metrics['path']
    link_fidelities = route_metrics['link_fidelities']
    hops = len(link_fidelities)

    step_results = []
    accumulated_p = 1.0

    for m in range(1, hops + 1):
        sub_path = path[:m + 1]
        sub_links = link_fidelities[:m]
        current_link_f = link_fidelities[m - 1]
        current_p = (4.0 * current_link_f - 1.0) / 3.0
        accumulated_p *= current_p

        theoretical_f = (1.0 + 3.0 * accumulated_p) / 4.0

        # Mô phỏng Qiskit cho đoạn tuyến con gồm m chặng
        sub_metrics = {
            'path': sub_path,
            'link_fidelities': sub_links,
            'hops': m
        }
        sim_res = simulate_route_fidelity_qiskit(sub_metrics, shots=shots)
        measured_f = sim_res['measured_fidelity']

        decay_from_initial = (link_fidelities[0] - theoretical_f) * 100
        step_decay = (step_results[-1]['theoretical_f'] - theoretical_f) * 100 if step_results else 0.0

        step_results.append({
            'step': m,
            'segment': f"{sub_path[0]} -> {sub_path[-1]}",
            'nodes': sub_path,
            'current_link_fidelity': current_link_f,
            'theoretical_f': theoretical_f,
            'measured_f': measured_f,
            'accumulated_p': accumulated_p,
            'step_decay_pct': step_decay,
            'total_decay_pct': decay_from_initial
        })

    return step_results


def plot_fidelity_decay_analysis(sp_steps, fa_steps, save_path='fidelity_decay_analysis.png'):
    r"""
    Vẽ 2 biểu đồ phân tích chuyên sâu về sự suy giảm Fidelity (Fidelity Decay):
    - Đồ thị 1: Diễn tiến suy giảm từng chặng trên Tuyến ngắn nhất vs Tuyến tối ưu Fidelity (Lý thuyết & Qiskit thực nghiệm).
    - Đồ thị 2: Khảo sát tổng quát tốc độ suy giảm theo số chặng (1 -> 8 chặng) với các cấp chất lượng kênh khác nhau,
               đánh dấu các ngưỡng tới hạn (Ngưỡng QKD 81%, Ngưỡng vướng lượng tử 50%).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    # -------------------------------------------------------------------------
    # ĐỒ THỊ 1: SUY GIẢM TỪNG CHẶNG TRÊN 2 TUYẾN THỰC TẾ
    # -------------------------------------------------------------------------
    sp_steps_num = [s['step'] for s in sp_steps]
    sp_theory = [s['theoretical_f'] * 100 for s in sp_steps]
    sp_meas = [s['measured_f'] * 100 for s in sp_steps]

    fa_steps_num = [s['step'] for s in fa_steps]
    fa_theory = [s['theoretical_f'] * 100 for s in fa_steps]
    fa_meas = [s['measured_f'] * 100 for s in fa_steps]

    # Vẽ Tuyến 1
    ax1.plot(sp_steps_num, sp_theory, 'r--', linewidth=2.2, label='Tuyến ngắn nhất (Lý thuyết)', alpha=0.85)
    ax1.plot(sp_steps_num, sp_meas, 'ro-', markersize=8, linewidth=2.0, label='Tuyến ngắn nhất (Qiskit Aer)')

    # Vẽ Tuyến 2
    ax1.plot(fa_steps_num, fa_theory, 'b--', linewidth=2.2, label='Tuyến Fidelity-Aware (Lý thuyết)', alpha=0.85)
    ax1.plot(fa_steps_num, fa_meas, 'bs-', markersize=8, linewidth=2.0, label='Tuyến Fidelity-Aware (Qiskit Aer)')

    # Đường ngưỡng an toàn QKD và ngưỡng vướng
    ax1.axhline(81.0, color='#ff7f0e', linestyle=':', linewidth=1.8, label='Ngưỡng an toàn QKD BB84 (81%)')
    ax1.axhline(50.0, color='#7f7f7f', linestyle=':', linewidth=1.8, label='Ngưỡng vướng lượng tử (50%)')

    ax1.set_title("Suy giảm Fidelity qua từng chặng thực tế\n(Tuyến ngắn nhất vs Tuyến Fidelity-Aware)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Chặng truyền dẫn tích lũy (Cumulative Hop)", fontsize=11)
    ax1.set_ylabel("Độ trung thực Fidelity (%)", fontsize=11)
    ax1.set_xticks(range(1, max(len(sp_steps), len(fa_steps)) + 1))
    ax1.set_ylim(45, 102)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9.2, frameon=True)

    # Thêm nhãn giá trị trên đồ thị 1
    for s in sp_steps:
        ax1.annotate(f"{s['measured_f']*100:.1f}%", (s['step'], s['measured_f']*100),
                     textcoords="offset points", xytext=(-15, -18), fontsize=9, color='#d62728', fontweight='bold')
    for s in fa_steps:
        ax1.annotate(f"{s['measured_f']*100:.1f}%", (s['step'], s['measured_f']*100),
                     textcoords="offset points", xytext=(8, 8), fontsize=9, color='#1f77b4', fontweight='bold')

    # -------------------------------------------------------------------------
    # ĐỒ THỊ 2: KHẢO SÁT QUY LUẬT SUY GIẢM TỔNG QUÁT THEO SỐ CHẶNG (1 - 8 chặng)
    # -------------------------------------------------------------------------
    hops_range = list(range(1, 9))
    qualities = [
        ('Kênh siêu sạch (F_link = 0.98)', 0.98, '#2ca02c'),
        ('Kênh cao cấp (F_link = 0.95)', 0.95, '#1f77b4'),
        ('Kênh tiêu chuẩn (F_link = 0.90)', 0.90, '#9467bd'),
        ('Kênh đô thị nhiễu (F_link = 0.82)', 0.82, '#d62728'),
    ]

    for label, f_val, color in qualities:
        p_val = (4.0 * f_val - 1.0) / 3.0
        f_decay = [((1.0 + 3.0 * (p_val ** n)) / 4.0) * 100 for n in hops_range]
        ax2.plot(hops_range, f_decay, marker='o', linewidth=2.0, label=label, color=color)

    ax2.axhline(81.0, color='#ff7f0e', linestyle=':', linewidth=1.8, label='Ngưỡng an toàn QKD (81%)')
    ax2.axhline(50.0, color='#7f7f7f', linestyle=':', linewidth=1.8, label='Ngưỡng vướng lượng tử (50%)')

    ax2.set_title("Mô hình suy giảm Fidelity tổng quát\ntheo số lượng chặng truyền dẫn (1 - 8 chặng)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Số lượng chặng (Number of Hops)", fontsize=11)
    ax2.set_ylabel("Độ trung thực cuối-cuối F_e2e (%)", fontsize=11)
    ax2.set_xticks(hops_range)
    ax2.set_ylim(25, 102)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9.2, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"-> Đã lưu biểu đồ phân tích suy giảm Fidelity tại: {os.path.abspath(save_path)}")


def run_fidelity_decay_evaluation():
    """
    Hàm thực thi phân tích và đánh giá mức độ suy giảm trạng thái lượng tử (Fidelity Decay):
    1. Lấy thông tin các tuyến đường từ cấu trúc mạng.
    2. Đánh giá chi tiết suy giảm qua từng chặng (Lý thuyết và Mô phỏng Qiskit Aer).
    3. Trình bày bảng theo dõi chi tiết từng bước.
    4. Xuất biểu đồ so sánh `fidelity_decay_analysis.png`.
    """
    print("=================================================================")
    print("   ĐÁNH GIÁ MỨC ĐỘ SUY GIẢM TRẠNG THÁI LƯỢNG TỬ (FIDELITY DECAY)")
    print("=================================================================\n")

    net = QuantumNetworkGraph.create_default_topology()
    sp_metrics = net.find_shortest_distance_route('Alice', 'Bob')
    fa_metrics = net.find_fidelity_aware_route('Alice', 'Bob')

    print("[1] PHÂN TÍCH SUY GIẢM DỌC THEO TUYẾN ĐƯỜNG NGẮN NHẤT (Alice -> R1 -> Bob):")
    print("-----------------------------------------------------------------------------------------")
    sp_steps = evaluate_stepwise_fidelity_decay(sp_metrics, shots=2000)
    print(f"{'Chặng':<7} | {'Đoạn truyền dẫn':<18} | {'F_link':<8} | {'F_lý thuyết':<13} | {'F_Qiskit':<10} | {'Độ suy giảm':<12}")
    print("-" * 80)
    for s in sp_steps:
        print(f"Chặng {s['step']:<1} | {s['segment']:<18} | {s['current_link_fidelity']:<8.2f} | "
              f"{s['theoretical_f']*100:<12.2f}% | {s['measured_f']*100:<9.2f}% | -{s['step_decay_pct']:<10.2f}%")
    print(f"=> Tổng suy giảm sau toàn tuyến: -{sp_steps[-1]['total_decay_pct']:.2f}% (Từ {sp_steps[0]['current_link_fidelity']*100:.1f}% xuống {sp_steps[-1]['theoretical_f']*100:.2f}%)")

    print("\n[2] PHÂN TÍCH SUY GIẢM DỌC THEO TUYẾN TỐI ƯU FIDELITY (Alice -> R2 -> R3 -> Bob):")
    print("-----------------------------------------------------------------------------------------")
    fa_steps = evaluate_stepwise_fidelity_decay(fa_metrics, shots=2000)
    print(f"{'Chặng':<7} | {'Đoạn truyền dẫn':<18} | {'F_link':<8} | {'F_lý thuyết':<13} | {'F_Qiskit':<10} | {'Độ suy giảm':<12}")
    print("-" * 80)
    for s in fa_steps:
        print(f"Chặng {s['step']:<1} | {s['segment']:<18} | {s['current_link_fidelity']:<8.2f} | "
              f"{s['theoretical_f']*100:<12.2f}% | {s['measured_f']*100:<9.2f}% | -{s['step_decay_pct']:<10.2f}%")
    print(f"=> Tổng suy giảm sau toàn tuyến: -{fa_steps[-1]['total_decay_pct']:.2f}% (Từ {fa_steps[0]['current_link_fidelity']*100:.1f}% xuống {fa_steps[-1]['theoretical_f']*100:.2f}%)")

    print("\n[3] XUẤT BIỂU ĐỒ PHÂN TÍCH SUY GIẢM (FIDELITY DECAY CURVES):")
    plot_file = 'fidelity_decay_analysis.png'
    plot_fidelity_decay_analysis(sp_steps, fa_steps, save_path=plot_file)

    print("\n=================================================================")
    print("                     KẾT LUẬN & ĐÁNH GIÁ CHUYÊN SÂU")
    print("=================================================================")
    print("1. TỐC ĐỘ SUY GIẢM (DECAY RATE):")
    print(f"   • Tuyến ngắn nhất (F_link ~ 0.82): Suy giảm rất dốc (-{sp_steps[1]['step_decay_pct']:.2f}% sau 1 lần BSM).")
    print(f"     Trạng thái tụt xuống còn {sp_steps[-1]['theoretical_f']*100:.2f}%, THẤP HƠN ngưỡng an toàn QKD (81%).")
    print(f"   • Tuyến Fidelity-Aware (F_link ~ 0.98): Suy giảm rất êm (-{fa_steps[1]['step_decay_pct']:.2f}% và -{fa_steps[2]['step_decay_pct']:.2f}%).")
    print(f"     Sau 3 chặng vẫn đạt {fa_steps[-1]['theoretical_f']*100:.2f}%, VƯỢT XA ngưỡng an toàn QKD.")
    print("\n2. GIỚI HẠN CHẶNG (MAX TRANSMISSION HOPS):")
    print("   • Với kênh F=0.82: Không thể truyền quá 1 chặng mà vẫn duy trì an toàn QKD.")
    print("     Ở chặng thứ 4, F tụt dưới 50% (mất trạng thái vướng hoàn toàn, trở thành trạng thái phân ly).")
    print("   • Với kênh F=0.98: Có thể truyền tới 8 chặng liên tiếp qua các trạm lặp mà vẫn đạt F > 83%.")
    print("=================================================================\n")

    return sp_steps, fa_steps


def run_routing_demonstration():
    """
    Hàm thực thi trình diễn định tuyến hoàn chỉnh:
    1. Khởi tạo cấu trúc liên kết mạng lượng tử 8 nút.
    2. Chạy Dijkstra Shortest Path vs. Fidelity-Aware Routing.
    3. In kết quả phân tích so sánh chi tiết.
    4. Tạo sơ đồ trực quan hóa đồ thị `quantum_routing_topology.png`.
    5. Chạy mô phỏng kiểm chứng Qiskit Aer với mô hình nhiễu trên cả 2 cơ sở.
    """
    print("=================================================================")
    print("     THỬ NGHIỆM ĐỊNH TUYẾN MẠNG LƯỢNG TỬ (QUANTUM ROUTING)")
    print("=================================================================\n")

    net = QuantumNetworkGraph.create_default_topology()

    print("[1] CẤU TRÚC MẠNG LƯỢNG TỬ ĐA NÚT:")
    print(f"  - Tổng số nút: {len(net.graph.nodes)} (Alice, Bob, R1, R2, R3, R4, R5, R6)")
    print(f"  - Tổng số liên kết: {len(net.graph.edges)}")
    print("  - Bảng thông số kênh lượng tử:")
    for u, v, data in net.graph.edges(data=True):
        print(f"    * {u:>5} <---> {v:<5}: d = {data['distance']:4.1f} km, F_link = {data['fidelity']:.2f}")

    # 1. Tìm đường ngắn nhất theo khoảng cách vật lý
    shortest_metrics = net.find_shortest_distance_route('Alice', 'Bob')

    # 2. Tìm đường tối ưu độ trung thực (Fidelity-Aware)
    fidelity_metrics = net.find_fidelity_aware_route('Alice', 'Bob')

    print("\n-----------------------------------------------------------------")
    print("[2] KẾT QUẢ THUẬT TOÁN TÌM ĐƯỜNG:")
    print("-----------------------------------------------------------------")
    print(f"▶ Tuyến 1: ĐƯỜNG NGẮN NHẤT (Shortest Distance Dijkstra)")
    print(f"  - Lộ trình: {' -> '.join(shortest_metrics['path'])}")
    print(f"  - Số chặng (Hops): {shortest_metrics['hops']}")
    print(f"  - Tổng khoảng cách vật lý: {shortest_metrics['total_distance_km']:.1f} km")
    print(f"  - Độ trung thực từng chặng: {[round(f, 3) for f in shortest_metrics['link_fidelities']]}")
    print(f"  - Độ trung thực lý thuyết cuối-cuối (F_e2e): {shortest_metrics['end_to_end_fidelity'] * 100:.2f}%")

    print(f"\n▶ Tuyến 2: ĐỊNH TUYẾN TỐI ƯU FIDELITY (Fidelity-Aware Routing)")
    print(f"  - Lộ trình: {' -> '.join(fidelity_metrics['path'])}")
    print(f"  - Số chặng (Hops): {fidelity_metrics['hops']}")
    print(f"  - Tổng khoảng cách vật lý: {fidelity_metrics['total_distance_km']:.1f} km")
    print(f"  - Độ trung thực từng chặng: {[round(f, 3) for f in fidelity_metrics['link_fidelities']]}")
    print(f"  - Độ trung thực lý thuyết cuối-cuối (F_e2e): {fidelity_metrics['end_to_end_fidelity'] * 100:.2f}%")

    # 3. Trực quan hóa đồ thị
    print("\n[3] XUẤT ĐỒ THỊ TRỰC QUAN HÓA:")
    plot_file = 'quantum_routing_topology.png'
    net.visualize_routes(shortest_metrics, fidelity_metrics, save_path=plot_file)

    # 4. Mô phỏng kiểm chứng trên Qiskit Aer với mô hình nhiễu
    print("\n[4] MÔ PHỎNG KIỂM CHỨNG BẰNG QISKIT AER (Đo trên cả 2 cơ sở Z & X):")
    print("-----------------------------------------------------------------")
    print("Đang mô phỏng Tuyến ngắn nhất...")
    sp_sim = simulate_route_fidelity_qiskit(shortest_metrics, shots=2000)
    print(f"  - Trùng khớp cơ sở Z: {sp_sim['same_z']*100:.2f}%, cơ sở X: {sp_sim['same_x']*100:.2f}%")
    print(f"  => Fidelity thực nghiệm (Measured): {sp_sim['measured_fidelity']*100:.2f}% (Lý thuyết: {shortest_metrics['end_to_end_fidelity']*100:.2f}%)")

    print("\nĐang mô phỏng Tuyến tối ưu Fidelity...")
    fa_sim = simulate_route_fidelity_qiskit(fidelity_metrics, shots=2000)
    print(f"  - Trùng khớp cơ sở Z: {fa_sim['same_z']*100:.2f}%, cơ sở X: {fa_sim['same_x']*100:.2f}%")
    print(f"  => Fidelity thực nghiệm (Measured): {fa_sim['measured_fidelity']*100:.2f}% (Lý thuyết: {fidelity_metrics['end_to_end_fidelity']*100:.2f}%)")

    print("\n=================================================================")
    print("                        KẾT LUẬN & ĐÁNH GIÁ")
    print("=================================================================")
    delta_fid = (fa_sim['measured_fidelity'] - sp_sim['measured_fidelity']) * 100
    print(f"• Tuyến Fidelity-Aware đi qua nhiều chặng hơn ({fidelity_metrics['hops']} chặng vs {shortest_metrics['hops']} chặng)")
    print(f"  và khoảng cách xa hơn ({fidelity_metrics['total_distance_km']:.1f} km vs {shortest_metrics['total_distance_km']:.1f} km).")
    print(f"• TUY NHIÊN, chất lượng vướng lượng tử cao hơn vượt trội: +{delta_fid:.2f}% Fidelity!")
    print(f"• Điều này chứng minh rằng trong Mạng Lượng tử, định tuyến theo khoảng cách vật lý")
    print(f"  KHÔNG ĐỒNG NGHĨA với định tuyến tối ưu chất lượng truyền thông lượng tử.")
    print("=================================================================\n")

    return shortest_metrics, fidelity_metrics


if __name__ == '__main__':
    run_routing_demonstration()
    print("\n" + "="*65)
    run_fidelity_decay_evaluation()

