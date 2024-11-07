import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy
import scipy.integrate


def draw_function(interval, linear_space_interval, function, samples,function_name = None):
    RIEMANN_SUM = scipy.integrate.quad(function,interval[0], interval[1])[0]
    print("SOMMA DI RIEMANN:",scipy.integrate.quad(function,interval[0],interval[1])[0])
    fig = plt.figure()
    fig.suptitle(f"{function_name}")

    x = np.linspace(linear_space_interval[0], linear_space_interval[1], samples)
    y = function(x)


    plt.grid()
    plt.axis(True)
    plt.axhline(y = 0, color = "black", lw = 1.2)
    plt.axvline(x = 0, color = "black", lw = 1.2)
    plt.plot(x,y)
    plt.legend([RIEMANN_SUM])


    interval_x = np.linspace(interval[0],interval[1],samples)
    interval_y = function(interval_x)



    full_range = [(interval[1],0)]
    for i in range(len(interval_x)):
        full_range.insert(1,(interval_x[i],interval_y[i]))
    full_range.append((interval[0],0))

    poly = mpatches.Polygon(full_range, facecolor = "orange", edgecolor = "gray", alpha = 0.2)
    plt.gca().add_patch(poly)


def create_riemann(interval, linear_space_interval, interval_density, function, number_graphs, samples, title = None, function_name = None):
    
    draw_function(interval, linear_space_interval, function, samples, function_name)

    fig = plt.figure()
    fig.suptitle(f"{title}({function_name})")
    
    x = np.linspace(linear_space_interval[0], linear_space_interval[1],samples)
    y = function(x)


    steps = {i:((1/pow(interval_density,i))) for i in range(number_graphs)}
    idx_plot = 1

    for i in range(len(steps)):

        sommaInferiore = 0
        sommaSuperiore = 0

        #disegnare la somma inferiore di riemann
        plt.subplot(len(steps),2,idx_plot)
        plt.title(f"Somma inferiore con dx={steps[i]}")
        
        plt.grid()
        plt.axis(True)

        plt.axhline(y = 0, lw = 1.2, color = "black")
        plt.axvline(x = 0, lw = 1.2, color = "black")


        
        step_range = np.arange(interval[0], interval[1], steps[i])
        plt.plot(x,y)
        for j in step_range:
            
            
            prev_p = j
            next_p = prev_p + steps[i]

            
            
            

            plt.plot(prev_p,function(prev_p), marker = "o", color = "black")

            rect_inferior = mpatches.Rectangle((prev_p,0), steps[i], function(prev_p), alpha = 0.2,
            facecolor = "red",
            edgecolor = "black",
            )
            

            plt.gca().add_patch(rect_inferior)
            sommaInferiore+=(function(prev_p)*steps[i])


        plt.legend([sommaInferiore])
        
        idx_plot+=1

        #disegnare la somma superiore di riemann
        plt.subplot(len(steps),2,idx_plot)

        plt.title(f"Somma superiore con dx={steps[i]}")
        plt.grid()
        plt.axis(True)

        plt.axhline(y = 0, lw = 1.2, color = "black")
        plt.axvline(x = 0, lw = 1.2, color = "black")
        
        step_range = np.arange(interval[0], interval[1], steps[i])
        
        plt.plot(x,y)
        for j in step_range:
            prev_p = j
            next_p = prev_p+steps[i]
            

           

            plt.plot(next_p,function(next_p), marker = "o", color = "black")

            rect_superior = mpatches.Rectangle((prev_p,0), steps[i], function(next_p), alpha = 0.2,
            facecolor = "green",
            edgecolor = "black",
            )
            
            
            plt.gca().add_patch(rect_superior) 
            sommaSuperiore+=(function(next_p)*steps[i])



        plt.legend([sommaSuperiore])
        idx_plot+=1
    plt.show()

    



def ln(x):
    return np.log(2*x + 7)

def quad(x):
    return -x**2 + 10*x - 12 

def sinus(x):
    return np.sin(x)


def main():
    INTERVAL_DENSITY = 2
    INTERVAL = [-2,9]
    LINEAR_SPACE_INTERVAL = [-5,10]
    SAMPLES = 10000
    NUMBER_GRAPHS = 3

    create_riemann(INTERVAL, LINEAR_SPACE_INTERVAL, INTERVAL_DENSITY, ln, NUMBER_GRAPHS, SAMPLES, "Somme di Riemann", "ln(2x+7)")

    create_riemann(INTERVAL, LINEAR_SPACE_INTERVAL, INTERVAL_DENSITY, quad, NUMBER_GRAPHS, SAMPLES, "Somme di Riemann", "(-x^2 + 10x - 12)")

    create_riemann(INTERVAL, LINEAR_SPACE_INTERVAL, INTERVAL_DENSITY, sinus, NUMBER_GRAPHS, SAMPLES, "Somme di Riemann", "sin(x)")



if __name__ == "__main__":
    main()

