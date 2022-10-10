import numpy as np
import ga


def main():
    # entradas da equação
    '''
    Saco de dormir; Peso: 15kg; Pontos 15
    Corda; Peso: 3kg; Pontos: 10
    Canivete; Peso: 2kg; Pontos: 10
    Tocha: Peso: 5kg; Pontos: 5
    Garrafa; Peso: 9kg; Pontos: 8
    Comida; Peso: 20kg; Pontos: 17
    
    '''
    
    equation_weights = [15,3,2,5,9,20]
    equation_points = [15,10,10,5,8,17]

    # número de pesos a otimizar
    num_weights = 6

    sol_per_pop = 8

    # população tem sol_per_pop cromossomos com num_weights gens
    pop_size = (sol_per_pop, num_weights)

    # População inicial
    new_population = np.random.randint(2, size=pop_size)

    # Algoritmo genético
    num_generations = 100
    num_parents_mating = 4

    for generation in range(num_generations):
        print(f"Geração: {generation}")

        # medir o ‘fitness’ de cada cromossomo na população
        fitness_points = ga.cal_pop_fitness(equation_weights,equation_points, new_population)

        print("Valores de fitness:")
        print(fitness_points)

        # Selecionar os melhores pais na população para o cruzamento
        parents = ga.select_mating_pool(new_population, fitness_points, num_parents_mating)

        print("Genitores selecionados:")
        print(parents)

        # formar a próxima geração usando crossover
        offspring_crossover = ga.crossover(parents, offspring_size=(
            pop_size[0] - parents.shape[0], num_weights
        ))
        print("Resultado do crossover:")
        print(offspring_crossover)

        # adicionar variações aos filhos usando mutação
        offspring_mutation = ga.mutation(offspring_crossover)
        print("Resultado da mutação:")
        print(offspring_mutation)

        # criar a nova população baseada nos pais e filhos
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        best_result = np.max(np.sum(new_population*equation_points, axis=1))
        print(f"Melhor resultado depois da geração {generation}: {best_result}")

    fitness = ga.cal_pop_fitness(equation_points,equation_weights, new_population)
    best_match_idx = np.where(fitness == np.max(fitness))

    print("Melhor solução: ", new_population[best_match_idx, :])
    print("Fitness da melhor solução: ", fitness[best_match_idx])


if __name__ == '__main__':
    main()
