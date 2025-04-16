import copy
import math
from multiprocessing import Manager, Pool
import concurrent.futures
import random
import numpy
from evaluadorBlosum import evaluadorBlosum 
from fastaReader import fastaReader     

class bacteria():
    def __init__(self, numBacterias):
        manager = Manager()
        self.blosumScore = manager.list(range(numBacterias))
        self.tablaAtract = manager.list(range(numBacterias))
        self.tablaRepel = manager.list(range(numBacterias))
        self.tablaInteraction = manager.list(range(numBacterias))
        self.tablaFitness = manager.list(range(numBacterias))
        self.granListaPares = manager.list(range(numBacterias))
        self.NFE = manager.list(range(numBacterias))
        # Parámetros para la adaptación dinámica del operador de quimiotaxis
        self.dynamic_maxSwim = 10
        self.dynamic_stepSize = 1
        self.prev_avg_fitness = None
        # Parámetros para el componente de chemiotaxis (atracción/repulsión)
        self.d_attr = 0.1
        self.w_attr = 0.2
        self.h_rep = self.d_attr
        self.w_rep = 10
        # Para reportar el número de evaluaciones parciales (opcional)
        self.parcialNFE = 0

    def resetListas(self, numBacterias):
        manager = Manager()
        self.blosumScore = manager.list(range(numBacterias))
        self.tablaAtract = manager.list(range(numBacterias))
        self.tablaRepel = manager.list(range(numBacterias))
        self.tablaInteraction = manager.list(range(numBacterias))
        self.tablaFitness = manager.list(range(numBacterias))
        self.granListaPares = manager.list(range(numBacterias))
        self.NFE = manager.list(range(numBacterias))

    def cuadra(self, numSec, poblacion):
        # Alinea las secuencias de cada candidato: para cada bacteria se iguala la longitud de todas las secuencias
        for i in range(len(poblacion)):
            bacterTmp = list(poblacion[i])
            bacterTmp = bacterTmp[:numSec]
            maxLen = 0
            for j in range(numSec):
                if len(bacterTmp[j]) > maxLen:
                    maxLen = len(bacterTmp[j])
                for t in range(numSec):
                    gap_count = maxLen - len(bacterTmp[t])
                    if gap_count > 0:
                        bacterTmp[t].extend(["-"] * gap_count)
                    poblacion[i] = tuple(bacterTmp)

    def limpiaColumnas(self):
        # (Método opcional: para eliminar columnas compuestas solo por gaps)
        i = 0
        while i < len(self.matrix.seqs[0]):
            if self.gapColumn(i):
                self.deleteCulmn(i)
            else:
                i += 1

    def deleteCulmn(self, pos):
        for i in range(len(self.matrix.seqs)):
            self.matrix.seqs[i] = self.matrix.seqs[i][:pos] + self.matrix.seqs[i][pos+1:]

    def gapColumn(self, col):
        for i in range(len(self.matrix.seqs)):
            if self.matrix.seqs[i][col] != "-":
                return False
        return True

    def tumbo(self, numSec, poblacion, numGaps):
        # Inserta gaps en posiciones aleatorias en cada bacteria
        for i in range(len(poblacion)):
            bacterTmp = list(poblacion[i])
            for j in range(numGaps):
                seqnum = random.randint(0, len(bacterTmp)-1)
                pos = random.randint(0, len(bacterTmp[seqnum]))
                part1 = bacterTmp[seqnum][:pos]
                part2 = bacterTmp[seqnum][pos:]
                temp = part1 + ["-"] + part2
                bacterTmp[seqnum] = temp
                poblacion[i] = tuple(bacterTmp)

    def creaGranListaPares(self, poblacion):
        # Genera la lista de pares únicos por candidato (para evaluación con Blosum)
        for i in range(len(poblacion)):
            pares = []
            bacterTmp = list(poblacion[i])
            for j in range(len(bacterTmp)):
                column = self.getColumn(bacterTmp, j)
                pares += self.obtener_pares_unicos(column)
            self.granListaPares[i] = pares

    def evaluaFila(self, fila, num):
        evaluador = evaluadorBlosum()  # Asegúrate de tener definido evaluadorBlosum
        score = 0
        for par in fila:
            score += evaluador.getScore(par[0], par[1])
        self.blosumScore[num] = score

    def evaluaBlosum(self):
        with Pool() as pool:
            args = [(copy.deepcopy(self.granListaPares[i]), i) for i in range(len(self.granListaPares))]
            pool.starmap(self.evaluaFila, args)

    def getColumn(self, bacterTmp, colNum):
        return [bacterTmp[i][colNum] for i in range(len(bacterTmp))]

    def obtener_pares_unicos(self, columna):
        pares_unicos = set()
        for i in range(len(columna)):
            for j in range(i+1, len(columna)):
                par = tuple(sorted([columna[i], columna[j]]))
                pares_unicos.add(par)
        return list(pares_unicos)

    def compute_diff(self, args):
        indexBacteria, otherBlosumScore, self_blosumScore, d, w = args
        diff = (self_blosumScore[indexBacteria] - otherBlosumScore) ** 2.0
        self.NFE[indexBacteria] += 1
        return d * numpy.exp(w * diff)

    def compute_cell_interaction(self, indexBacteria, d, w, atracTrue):
        with Pool() as pool:
            args = [(indexBacteria, otherBlosumScore, self.blosumScore, d, w) for otherBlosumScore in self.blosumScore]
            results = pool.map(self.compute_diff, args)
            pool.close()
            pool.join()
        total = sum(results)
        if atracTrue:
            self.tablaAtract[indexBacteria] = total
        else:
            self.tablaRepel[indexBacteria] = total

    def creaTablaAtract(self, poblacion, d, w):
        for indexBacteria in range(len(poblacion)):
            self.compute_cell_interaction(indexBacteria, d, w, True)

    def creaTablaRepel(self, poblacion, d, w):
        for indexBacteria in range(len(poblacion)):
            self.compute_cell_interaction(indexBacteria, d, w, False)

    def creaTablasAtractRepel(self, poblacion, dAttr, wAttr, dRepel, wRepel):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.creaTablaAtract, poblacion, dAttr, wAttr)
            executor.submit(self.creaTablaRepel, poblacion, dRepel, wRepel)

    def creaTablaInteraction(self):
        for i in range(len(self.tablaAtract)):
            self.tablaInteraction[i] = self.tablaAtract[i] + self.tablaRepel[i]

    def creaTablaFitness(self):
        for i in range(len(self.tablaInteraction)):
            self.tablaFitness[i] = self.blosumScore[i] + self.tablaInteraction[i]

    def getNFE(self):
        return sum(self.NFE)

    def obtieneBest(self, globalNFE):
        bestIdx = 0
        for i in range(len(self.tablaFitness)):
            if self.tablaFitness[i] > self.tablaFitness[bestIdx]:
                bestIdx = i
        print("-------------------   Best: ", bestIdx,
              " Fitness: ", self.tablaFitness[bestIdx],
              " BlosumScore: ", self.blosumScore[bestIdx],
              " Interaction: ", self.tablaInteraction[bestIdx],
              " NFE: ", globalNFE)
        return bestIdx, self.tablaFitness[bestIdx]

    def replaceWorst(self, poblacion, best):
        worst = 0
        for i in range(len(self.tablaFitness)):
            if self.tablaFitness[i] < self.tablaFitness[worst]:
                worst = i
        poblacion[worst] = copy.deepcopy(poblacion[best])

    # ----------------- Operador de quimiotaxis ----------------- #
    def moverPequeno(self, bacterium, stepSize):
        candidate = copy.deepcopy(bacterium)
        seq_index = random.randint(0, len(candidate)-1)
        if len(candidate[seq_index]) < 2:
            return candidate
        pos = random.randint(1, len(candidate[seq_index]) - 2)
        direction = random.choice([-1, 1])
        for _ in range(stepSize):
            neighbor = pos + direction
            if neighbor < 0 or neighbor >= len(candidate[seq_index]):
                break
            candidate[seq_index][pos], candidate[seq_index][neighbor] = candidate[seq_index][neighbor], candidate[seq_index][pos]
            pos = neighbor
        return candidate

    def evaluaCandidate(self, candidate):
        granPares = []
        for j in range(len(candidate[0])):
            column = self.getColumn(candidate, j)
            granPares.extend(self.obtener_pares_unicos(column))
        evaluador = evaluadorBlosum()  # Asegúrate de tener definido evaluadorBlosum
        score = 0
        for par in granPares:
            score += evaluador.getScore(par[0], par[1])
        return score

    def quimiotaxis(self, numSec, poblacion, maxSwim, stepSize):
        for idx in range(len(poblacion)):
            current_candidate = poblacion[idx]
            current_fitness = self.evaluaCandidate(current_candidate)
            swim_counter = 0
            while swim_counter < maxSwim:
                candidate = self.moverPequeno(current_candidate, stepSize)
                candidate_fitness = self.evaluaCandidate(candidate)
                if candidate_fitness > current_fitness:
                    current_candidate = candidate
                    current_fitness = candidate_fitness
                    swim_counter += 1
                else:
                    break
            poblacion[idx] = current_candidate

    def update_parameters(self, poblacion):
        avg_fitness = sum(self.tablaFitness) / len(self.tablaFitness)
        if self.prev_avg_fitness is not None:
            if avg_fitness > self.prev_avg_fitness:
                self.dynamic_maxSwim = int(self.dynamic_maxSwim * 1.05)
                self.dynamic_stepSize = int(self.dynamic_stepSize * 1.05)
            else:
                self.dynamic_maxSwim = max(1, int(self.dynamic_maxSwim * 0.95))
                self.dynamic_stepSize = max(1, int(self.dynamic_stepSize * 0.95))
        self.prev_avg_fitness = avg_fitness
        print("Parámetros actualizados: maxSwim =", self.dynamic_maxSwim,
              "stepSize =", self.dynamic_stepSize)

    def aplicar_elitismo(self, poblacion):
        indices = list(range(len(poblacion)))
        indices.sort(key=lambda i: self.tablaFitness[i], reverse=True)
        elite_size = max(2, int(len(poblacion) * 0.1))
        elite_indices = indices[:elite_size]
        elite = [poblacion[i] for i in elite_indices]
        print("Elite seleccionada (índices):", elite_indices)
        return elite

    def recombinar_bacterias(self, bact1, bact2):
        nueva_bacteria = []
        for seq1, seq2 in zip(bact1, bact2):
            punto_cruce = random.randint(0, len(seq1))
            nueva_seq = seq1[:punto_cruce] + seq2[punto_cruce:]
            nueva_bacteria.append(nueva_seq)
        return nueva_bacteria

    def quimiotaxis_con_adaptacion(self, numSec, poblacion):
        for idx in range(len(poblacion)):
            current_candidate = poblacion[idx]
            current_fitness = self.evaluaCandidate(current_candidate)
            swim_counter = 0
            while swim_counter < self.dynamic_maxSwim:
                candidate = self.moverPequeno(current_candidate, self.dynamic_stepSize)
                candidate_fitness = self.evaluaCandidate(candidate)
                if candidate_fitness > current_fitness:
                    current_candidate = candidate
                    current_fitness = candidate_fitness
                    swim_counter += 1
                else:
                    break
            poblacion[idx] = current_candidate
        self.update_parameters(poblacion)
        bestIdx, bestFitness = self.obtieneBest(sum(self.NFE))
        self.replaceWorst(poblacion, bestIdx)

    # --------------- Métodos adicionales de Chemiotaxis --------------- #
    def compute_cell_interaction(self, bacterium, poblacion, d, w):
        total = 0.0
        for other in poblacion:
            diff = (bacterium.blosumScore - other.blosumScore) ** 2.0
            total += d * math.exp(w * diff)
        return total

    def attract_repel(self, bacterium, poblacion):
        attract = self.compute_cell_interaction(bacterium, poblacion, -self.d_attr, -self.w_attr)
        repel   = self.compute_cell_interaction(bacterium, poblacion, self.h_rep, -self.w_rep)
        return attract + repel

    def chemio(self, bacterium, poblacion):
        bacterium.interaction = self.attract_repel(bacterium, poblacion)
        bacterium.fitness = bacterium.blosumScore + bacterium.interaction

    def doChemioTaxis(self, poblacion):
        self.parcialNFE = 0
        for bacterium in poblacion:
            self.chemio(bacterium, poblacion)
            self.parcialNFE += bacterium.NFE
            bacterium.NFE = 0

    def reducir_poblacion(self, poblacion):
        reduction_rate = 0.1
        reduce_by = int(len(poblacion) * reduction_rate)
        poblacion.sort(key=lambda x: x.fitness)
        for _ in range(reduce_by):
            poblacion.pop(0)

    def generar_nuevas_bacterias(self, seqs, elite, num_new_bacterias):
        nuevas_bacterias = []
        for _ in range(num_new_bacterias):
            if len(elite) >= 2:
                parent1, parent2 = random.sample(elite, 2)
            elif len(elite) == 1:
                parent1 = parent2 = elite[0]
            else:
                parent1, parent2 = random.sample(elite, 2)
            # Se asume que cada bacteria tiene un atributo "matrix.seqs"; aquí se utiliza la recombinación de las secuencias
            new_bacteria = self.recombinar_bacterias(parent1.matrix.seqs, parent2.matrix.seqs)
            nuevas_bacterias.append(new_bacteria)
        return nuevas_bacterias

    def randomBacteria(self, seqs):
        # Se espera que la clase bacteria pueda inicializarse con "seqs" (por ejemplo, leídas de un archivo FASTA)
        bact = bacteria(seqs)
        bact.tumboNado(random.randint(1, 10))
        return bact

    def insertRamdomBacterias(self, seqs, num, poblacion):
        for _ in range(num):
            poblacion.append(self.randomBacteria(seqs))
            poblacion.sort(key=lambda x: x.fitness)
            del poblacion[0]
