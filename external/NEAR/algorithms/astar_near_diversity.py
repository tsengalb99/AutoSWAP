import copy
import time
import zss

import dsl
from .core import ProgramLearningAlgorithm, ProgramNodeFrontier
from program_graph import ProgramGraph
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train, prep_execute_and_train_dset


def ignore_module_comp_fn(x, y):
    no_cost = set([
        'AtomToAtomModule',
        'AtomToListModule',
        'ListToListModule',
        'ListToAtomModule',
    ])
    if x in no_cost or y in no_cost:
        return 0
    else:
        return 1 if x != y else 0


def create_zss_graph(program):
    # takes very little time, not worth running in own proc
    if not isinstance(program, dsl.LibraryFunction):
        return zss.Node(program.name), 1
    else:
        current = zss.Node(program.name)
        total_nodes = 1
        for submodule, functionclass in program.submodules.items():
            node, ct = create_zss_graph(functionclass)
            current.addkid(node)
            total_nodes += ct
        return current, total_nodes


class ASTAR_NEAR_DIVERSITY(ProgramLearningAlgorithm):
    def __init__(self, frontier_capacity=float('inf'), existing_progs=[], q=lambda x: x**2):
        self.frontier_capacity = frontier_capacity
        self.existing_progs = [create_zss_graph(prog)[0] for prog in existing_progs]
        self.use_diversity = len(self.existing_progs) > 0
        self.q = q

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training root program ...")
        current = copy.deepcopy(graph.root_node)

        prepped_trainset, prepped_validset = prep_execute_and_train_dset(trainset, validset, device)

        initial_score = execute_and_train(current.program,
                                          prepped_validset,
                                          prepped_trainset,
                                          train_config,
                                          graph.output_type,
                                          graph.output_size,
                                          neural=True,
                                          device=device)
        log_and_print(
            "Initial training complete. Score from program is {:.4f} \n".format(1 - initial_score))

        order = 0
        frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
        frontier.add((float('inf'), order, current))
        num_children_trained = 0
        start_time = time.time()

        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []

        while len(frontier) != 0:
            current_f_score, _, current = frontier.pop(0)
            log_and_print("CURRENT program has fscore {:.4f}: {}".format(
                current_f_score, print_program(current.program, ignore_constants=(not verbose))))
            log_and_print("Current depth of program is {}".format(current.depth))
            log_and_print("Creating children for current node/program")
            children_nodes = graph.get_all_children(current)
            # prune if more than self.max_num_children
            if len(children_nodes) > graph.max_num_children:
                children_nodes = random.sample(
                    children_nodes, k=graph.max_num_children)  # sample without replacement
            log_and_print("{} total children to train for current node".format(len(children_nodes)))

            for child_node in children_nodes:
                child_start_time = time.time()

                log_and_print("Training child program: {}".format(
                    print_program(child_node.program, ignore_constants=(not verbose))))
                is_neural = not graph.is_fully_symbolic(child_node.program)
                child_node.score = execute_and_train(child_node.program,
                                                     prepped_validset,
                                                     prepped_trainset,
                                                     train_config,
                                                     graph.output_type,
                                                     graph.output_size,
                                                     neural=is_neural,
                                                     device=device)
                log_and_print("Time to train child {:.3f}".format(time.time() - child_start_time))
                num_children_trained += 1
                log_and_print("{} total children trained".format(num_children_trained))
                child_node.parent = current
                child_node.children = []
                order -= 1
                child_node.order = order  # insert order of exploration as tiebreaker for equivalent f-scores
                current.children.append(child_node)

                # Compute admissible structural diversity heuristic if needed
                structural_cost = 0
                if self.use_diversity:
                    child_prog_zss, child_node_ct = create_zss_graph(child_node.program)
                    for existing_prog in self.existing_progs:
                        triangle_dist = zss.simple_distance(child_prog_zss,
                                                            existing_prog,
                                                            label_dist=ignore_module_comp_fn)
                        if is_neural:
                            triangle_dist += graph.max_depth - child_node.depth
                        structural_cost += triangle_dist / len(self.existing_progs)
                    structural_cost = 1 / self.q(structural_cost)

                print(f'STRUCTURAL COST {structural_cost}')

                # computing path costs (f_scores)
                child_f_score = child_node.cost + child_node.score + structural_cost  # cost + heuristic

                log_and_print("DEBUG: f-score {}".format(child_f_score))

                if not is_neural and child_f_score < best_total_cost:
                    best_program = copy.deepcopy(child_node.program)
                    best_total_cost = child_f_score
                    best_programs_list.append({
                        "program": best_program,
                        "struct_cost": child_node.cost,
                        "score": child_node.score,
                        "path_cost": child_f_score,
                        "time": time.time() - start_time,
                        "parent_path": child_node.full_parent_path
                    })
                    log_and_print("New BEST program found:")
                    print_program_dict(best_programs_list[-1])

                if is_neural:
                    assert child_node.depth < graph.max_depth
                    child_tuple = (child_f_score, order, child_node)
                    frontier.add(child_tuple)

            # clean up frontier
            frontier.sort(tup_idx=0)
            while len(frontier) > 0 and frontier.peek(-1)[0] > best_total_cost:
                frontier.pop(-1)
            log_and_print("Frontier length is: {}".format(len(frontier)))
            log_and_print("Total time elapsed is {:.3f}".format(time.time() - start_time))

        if best_program is None:
            log_and_print("ERROR: no program found")

        return best_programs_list