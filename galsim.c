#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

typedef struct {
    double * restrict x;
    double * restrict y;
    double * restrict m;
    double * restrict vx;
    double * restrict vy;
    double * restrict brightness;
} Particle;

static double get_wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

void read_file(const char* filename, Particle* particles, int N) {
    FILE *file = fopen(filename, "rb");
    if (!file) { perror("fopen"); exit(1); }

    for (int i = 0; i < N; i++) {
        if (fread(&particles->x[i],          sizeof(double), 1, file) != 1 ||
            fread(&particles->y[i],          sizeof(double), 1, file) != 1 ||
            fread(&particles->m[i],          sizeof(double), 1, file) != 1 ||
            fread(&particles->vx[i],         sizeof(double), 1, file) != 1 ||
            fread(&particles->vy[i],         sizeof(double), 1, file) != 1 ||
            fread(&particles->brightness[i], sizeof(double), 1, file) != 1) {
            fprintf(stderr, "Error reading file\n");
            exit(1);
        }
    }
    fclose(file);
}

void write_file(const char* filename, Particle* particles, int N) {
    FILE *file = fopen(filename, "wb");
    if (!file) { perror("fopen"); exit(1); }

    for (int i = 0; i < N; i++) {
        fwrite(&particles->x[i],          sizeof(double), 1, file);
        fwrite(&particles->y[i],          sizeof(double), 1, file);
        fwrite(&particles->m[i],          sizeof(double), 1, file);
        fwrite(&particles->vx[i],         sizeof(double), 1, file);
        fwrite(&particles->vy[i],         sizeof(double), 1, file);
        fwrite(&particles->brightness[i], sizeof(double), 1, file);
    }
    fclose(file);
}

void simulate_profiled(Particle* __restrict__ particles, int N, double G, double dt, double eps,
                       double *t_alloc, double *t_force, double *t_update, double *t_free)
{
    double t0, t1;

    t0 = get_wall_seconds();
    double *ax = calloc(N, sizeof(double));
    double *ay = calloc(N, sizeof(double));
    t1 = get_wall_seconds();
    *t_alloc += (t1 - t0);

    t0 = get_wall_seconds();
    for (int i = 0; i < N; i++) {
        double axi = 0.0, ayi = 0.0;
        double mi = particles->m[i];
        for (int j = i + 1; j < N; j++) {
            double dx = particles->x[i] - particles->x[j];
            double dy = particles->y[i] - particles->y[j];
            double rij = sqrt(dx*dx + dy*dy);
            double reps = rij + eps;
            double a = G / (reps*reps*reps);

            axi   -= a * particles->m[j] * dx;
            ayi   -= a * particles->m[j] * dy;
            ax[j] += a * mi * dx;
            ay[j] += a * mi * dy;
        }
        ax[i] += axi;
        ay[i] += ayi;
    }
    t1 = get_wall_seconds();
    *t_force += (t1 - t0);

    t0 = get_wall_seconds();
    for (int i = 0; i < N; i++) {
        particles->vx[i] += dt * ax[i];
        particles->vy[i] += dt * ay[i];
        particles->x[i]  += dt * particles->vx[i];
        particles->y[i]  += dt * particles->vy[i];
    }
    t1 = get_wall_seconds();
    *t_update += (t1 - t0);

    t0 = get_wall_seconds();
    free(ax);
    free(ay);
    t1 = get_wall_seconds();
    *t_free += (t1 - t0);
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s N filename nsteps delta_t graphics\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    const char* filename = argv[2];
    int nsteps = atoi(argv[3]);
    double dt = atof(argv[4]);
    int graphics = atoi(argv[5]);

    if (graphics) {
        printf("Graphics enabled (should be OFF when timing)\n");
    }

    double G = 100.0 / N;
    double eps = 1e-3;

    Particle particles;
    particles.x          = malloc(N * sizeof(double));
    particles.y          = malloc(N * sizeof(double));
    particles.m          = malloc(N * sizeof(double));
    particles.vx         = malloc(N * sizeof(double));
    particles.vy         = malloc(N * sizeof(double));
    particles.brightness = malloc(N * sizeof(double));

    read_file(filename, &particles, N);

    double t_alloc=0, t_force=0, t_update=0, t_free=0;

    double start = get_wall_seconds();
    for (int i = 0; i < nsteps; i++) {
        simulate_profiled(&particles, N, G, dt, eps, &t_alloc, &t_force, &t_update, &t_free);
    }
    double end = get_wall_seconds();

    write_file("result.gal", &particles, N);

    printf("%d %f\n", N, end - start);
    printf("Total:  %f s\n", end - start);
    printf("alloc:  %f s\n", t_alloc);
    printf("force:  %f s\n", t_force);
    printf("update: %f s\n", t_update);
    printf("free:   %f s\n", t_free);

    free(particles.x); free(particles.y); free(particles.m);
    free(particles.vx); free(particles.vy); free(particles.brightness);

    return 0;
}

//kör med
// ./galsim 3000 input_data/ellipse_N_03000.gal 100 1e-5 0