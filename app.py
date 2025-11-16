from flask import Flask, render_template, request, redirect, send_file, session, url_for
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = '89033838145'


class Attr:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def val(self):
        return float(self.value)


class F:
    def __init__(self, a, b, c, d, L):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.L = L

    def calc(self, x):
        # Гарантируем неотрицательность, но сохраняем нелинейность
        result = self.a * (x ** 3) + self.b * (x ** 2) + self.c * x + self.d
        return max(0, min(1, result))  # Ограничиваем между 0 и 1


# Инициализация данных
v0 = {
    'L₁': Attr('летальность (mortality)', 0.1),
    'L₂': Attr('численность инфицированных (number of infected people)', 0.3),
    'L₃': Attr('численность населения региона (the number of civilizations)', 0.9),
    'L₄': Attr('численность госпитализированных (the number of hospitalized patients)', 0.15),
    'L₅': Attr('изолированность (isolation)', 0.2),
    'L₆': Attr('скорость распространения (propagation speed)', 0.5),
    'L₇': Attr('доступность лекарства (drug availability)', 0.4),
    'L₈': Attr('тяжесть симптомов (severity of symptoms)', 0.35),
    'L₉': Attr('количество умерших от заболевания (the number of deaths from the disease)', 0.05),
    'L₁₀': Attr('уровень медицины (level of medicine)', 0.6),
    'L₁₁': Attr('длительность инкубационного периода (duration of the incubation period)', 0.4),
    'L₁₂': Attr(
        'длительность периода полного развития болезни (duration of the period of full development of the disease)',
        0.45),
    'L₁₃': Attr('длительность реабилитационного периода (duration of the rehabilitation period)', 0.5),
    'L₁₄': Attr('устойчивость вируса к лекарствам (drug resistance of the virus)', 0.25),
    'L₁₅': Attr('степень осложнений заболевания (the degree of complications of the disease)', 0.3)
}

c = {
    'L₁*': Attr('летальность (mortality)', 1.0),
    'L₂*': Attr('численность инфицированных (number of infected people)', 1.0),
    'L₃*': Attr('численность населения региона (the number of civilizations)', 1.0),
    'L₄*': Attr('численность госпитализированных (the number of hospitalized patients)', 1.0),
    'L₅*': Attr('изолированность (isolation)', 1.0),
    'L₆*': Attr('скорость распространения (propagation speed)', 1.0),
    'L₇*': Attr('доступность лекарства (drug availability)', 1.0),
    'L₈*': Attr('тяжесть симптомов (severity of symptoms)', 1.0),
    'L₉*': Attr('количество умерших от заболевания (the number of deaths from the disease)', 1.0),
    'L₁₀*': Attr('уровень медицины (level of medicine)', 1.0),
    'L₁₁*': Attr('длительность инкубационного периода (duration of the incubation period)', 1.0),
    'L₁₂*': Attr(
        'длительность периода полного развития болезни (duration of the period of full development of the disease)',
        1.0),
    'L₁₃*': Attr('длительность реабилитационного периода (duration of the rehabilitation period)', 1.0),
    'L₁₄*': Attr('устойчивость вируса к лекарствам (drug resistance of the virus)', 1.0),
    'L₁₅*': Attr('степень осложнений заболевания (the degree of complications of the disease)', 1.0)
}

# Более интересные нелинейные функции
f = {
    'f₁': F(0.1, -0.2, 0.8, 0, "L₃"),      # Нелинейная зависимость от населения
    'f₂': F(0.2, -0.3, 0.9, 0, "L₁"),      # Зависимость от летальности
    'f₃': F(0.1, -0.1, 0.7, 0.1, "L₄"),    # Зависимость от госпитализированных
    'f₄': F(0.3, -0.4, 1.0, 0, "L₆"),      # Сильная зависимость от скорости распространения
    'f₅': F(0.2, -0.2, 0.8, 0, "L₇"),      # Зависимость от доступности лекарств
    'f₆': F(0.1, -0.1, 0.6, 0.2, "L₆"),    # Другая зависимость от скорости
    'f₇': F(0.4, -0.5, 1.1, 0, "L₁₀"),     # Зависимость от уровня медицины
    'f₈': F(0.3, -0.3, 0.7, 0, "L₁₄"),     # Зависимость от устойчивости вируса
    'f₉': F(0.2, -0.2, 0.8, 0, "L₅"),      # Зависимость от изолированности
    'f₁₀': F(0.4, -0.6, 1.2, 0, "L₂"),     # Сильная зависимость от числа инфицированных
    'f₁₁': F(0.1, -0.1, 0.5, 0.3, "L₆"),   # Зависимость от скорости
    'f₁₂': F(0.3, -0.4, 1.0, 0, "L₂"),     # Зависимость от числа инфицированных
    'f₁₃': F(0.2, -0.2, 0.6, 0.1, "L₁₄"),  # Зависимость от устойчивости
    'f₁₄': F(0.5, -0.7, 1.3, 0, "L₂"),     # Сильная нелинейная зависимость
    'f₁₅': F(0.1, -0.1, 0.4, 0.4, "L₁₃"),  # Зависимость от реабилитации
    'f₁₆': F(0.2, -0.3, 0.9, 0, "L₁₅"),    # Зависимость от осложнений
    'f₁₇': F(0.4, -0.5, 1.1, 0, "L₉"),     # Зависимость от умерших
    'f₁₈': F(0.1, -0.1, 0.3, 0.5, "L₁₃"),  # Зависимость от реабилитации
    'f₁₉': F(0.3, -0.4, 1.0, 0, "L₁₅"),    # Зависимость от осложнений
    'f₂₀': F(0.2, -0.2, 0.7, 0, "L₁"),     # Зависимость от летальности
    'f₂₁': F(0.5, -0.6, 1.2, 0, "L₇"),     # Сильная зависимость от лекарств
    'f₂₂': F(0.1, -0.1, 0.6, 0.2, "L₄"),   # Зависимость от госпитализированных
    'f₂₃': F(0.2, -0.3, 0.8, 0, "L₁₂"),    # Зависимость от периода развития
    'f₂₄': F(0.1, -0.1, 0.5, 0.3, "L₁₃"),  # Зависимость от реабилитации
    'f₂₅': F(0.4, -0.5, 1.1, 0, "L₇"),     # Зависимость от лекарств
    'f₂₆': F(0.3, -0.4, 1.0, 0, "L₈"),     # Зависимость от тяжести симптомов
    'f₂₇': F(0.2, -0.2, 0.7, 0, "L₁₂"),    # Зависимость от периода развития
    'f₂₈': F(0.5, -0.6, 1.2, 0, "L₉")      # Сильная зависимость от умерших
}


def q1(t):
    """Вероятность заражения - изменяется во времени"""
    return 0.3 + 0.2 * np.sin(2 * np.pi * t + 0.5)


def q2(t):
    """Вероятность госпитализации"""
    return 0.2 + 0.15 * np.cos(2 * np.pi * t + 1.0)


def q3(t):
    """Вероятность выздоровления"""
    return 0.4 + 0.25 * np.sin(2 * np.pi * t + 2.0)


def q4(t):
    """Вероятность летального исхода"""
    return 0.1 + 0.08 * np.cos(2 * np.pi * t + 1.5)


def q5(t):
    """Вероятность изоляции"""
    return 0.3 + 0.2 * np.sin(2 * np.pi * t + 0.8)


def non_negative(u):
    """Гарантирует неотрицательность всех компонент"""
    return [max(0.001, x) for x in u]  # Минимальное значение 0.001 вместо 0


def du_dt(u, t):
    # Распаковка переменных
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15 = u

    # Формулы из файла
    dl1_dt = (1 / c["L₁*"].val()) * (
        q4(t) - f['f₁'].calc(l3) * (q1(t) + q2(t) + q4(t) + q5(t))
    )

    dl2_dt = (1 / c["L₂*"].val()) * (
        f['f₂'].calc(l1) * f['f₃'].calc(l4) * f['f₄'].calc(l6)
        - f['f₅'].calc(l7) * (q1(t) + q4(t))
    )

    dl3_dt = (1 / c["L₃*"].val()) * (
        f['f₆'].calc(l6) * f['f₇'].calc(l10) * f['f₈'].calc(l14) * (q2(t) + q5(t))
        - f['f₉'].calc(l5)
    )

    dl4_dt = (1 / c["L₄*"].val()) * (
        q4(t) + q5(t)
    )

    dl5_dt = (1 / c["L₅*"].val()) * (
        (q4(t) + q5(t)) - (f['f₁₀'].calc(l2) * f['f₁₁'].calc(l6))
    )

    dl6_dt = (1 / c["L₆*"].val()) * (
        f['f₁₂'].calc(l2) * q5(t) - (q1(t) + q4(t))
    )

    dl7_dt = (1 / c["L₇*"].val()) * (
        f['f₁₃'].calc(l14) * (q2(t) + q4(t)) -
        (f['f₁₄'].calc(l2) * f['f₁₅'].calc(l13) * f['f₁₆'].calc(l15))
    )

    dl8_dt = (1 / c["L₈*"].val()) * (
        f['f₁₇'].calc(l9) * f['f₁₈'].calc(l13) * f['f₁₉'].calc(l15)
        - q1(t)
    )

    dl9_dt = (1 / c["L₉*"].val()) * (
        f['f₂₀'].calc(l1)
    )

    dl10_dt = (1 / c["L₁₀*"].val()) * (
        f['f₂₁'].calc(l7) * (q1(t) + q2(t) + q3(t) + q4(t) + q5(t))
        - f['f₂₂'].calc(l4) * f['f₂₃'].calc(l12) * f['f₂₄'].calc(l13)
    )

    dl11_dt = (1 / c["L₁₁*"].val()) * (
        q1(t)
    )

    dl12_dt = (1 / c["L₁₂*"].val()) * (
        -q1(t) * f['f₂₅'].calc(l7)
    )

    dl13_dt = (1 / c["L₁₃*"].val()) * (
        -q1(t)
    )

    dl14_dt = (1 / c["L₁₄*"].val()) * (
        f['f₂₆'].calc(l8) * f['f₂₇'].calc(l12) - q2(t)
    )

    dl15_dt = (1 / c["L₁₅*"].val()) * (
        f['f₂₈'].calc(l9) - q1(t)
    )

    return [
        dl1_dt, dl2_dt, dl3_dt, dl4_dt, dl5_dt,
        dl6_dt, dl7_dt, dl8_dt, dl9_dt, dl10_dt,
        dl11_dt, dl12_dt, dl13_dt, dl14_dt, dl15_dt
    ]



# --- вставь ниже определений функций q1..q5, non_negative, du_dt и перед блоком if __name__ == '__main__' ---

def generate_plots_from_form(form, want_main_plot=True, want_polar=False):
    """
    form: request.form (или похожий dict)
    want_main_plot: строить ли основной график (динамика)
    want_polar: строить ли полярные диаграммы (возвращает polar_plot_data)
    Возвращает tuple (plot_data_or_None, polar_plot_data_or_None)
    """
    # Обновляем параметры v0
    for key in v0:
        # если в форме нет поля — оставляем старое значение
        try:
            v0[key].value = float(form.get(f'v0_{key}', v0[key].value))
        except Exception:
            pass

    # Обновляем параметры f(x)
    for key in f:
        try:
            f[key].a = float(form.get(f'f_{key}_a', f[key].a))
            f[key].b = float(form.get(f'f_{key}_b', f[key].b))
            f[key].c = float(form.get(f'f_{key}_c', f[key].c))
            f[key].d = float(form.get(f'f_{key}_d', f[key].d))
        except Exception:
            pass

    plot_data = None
    polar_plot_data = None

    # Начальное состояние
    t0 = [v0[key].value for key in
          ['L₁','L₂','L₃','L₄','L₅','L₆','L₇','L₈','L₉','L₁₀','L₁₁','L₁₂','L₁₃','L₁₄','L₁₅']]

    # Построение основного графика (динамики)
    if want_main_plot:
        t_span_main = np.arange(0.0, 1.0 + 1e-9, 0.05)
        sol = odeint(du_dt, t0, t_span_main)
        sol = np.maximum(sol, 0.001)

        fig1, ax1 = plt.subplots(figsize=(14, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, 15))
        labels = ['L₁','L₂','L₃','L₄','L₅','L₆','L₇','L₈','L₉','L₁₀','L₁₁','L₁₂','L₁₃','L₁₄','L₁₅']

        for i in range(15):
            y = sol[:, i]
            ax1.plot(t_span_main, y, color=colors[i], linewidth=2)

            # --- ДОБАВЛЯЕМ ПОДПИСЬ В КОНЦЕ ЛИНИИ ---
            ax1.text(
                t_span_main[-1] + 0.01,   # Чуть правее графика
                y[-1],                    # Высота последней точки
                labels[i],                # Текст: L₁, L₂, ...
                fontsize=10,
                color='black',
                va='center'
            )


        ax1.set_xlabel('Время', fontsize=12)
        ax1.set_ylabel('Значения', fontsize=12)
        ax1.set_title('Динамика эпидемиологических параметров', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        ax1.set_xlim([0, 1.05])

        plt.tight_layout()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', bbox_inches='tight', dpi=100)
        buf1.seek(0)
        plot_data = base64.b64encode(buf1.getvalue()).decode('utf-8')
        plt.close(fig1)

    # Построение полярных диаграмм
    if want_polar:
        # читаем norm_bounds из формы
        norm_bounds = []
        for i in range(15):
            try:
                nb = float(form.get(f'norm_bound_{i}', 1.0))
            except Exception:
                nb = 1.0
            norm_bounds.append(nb)

        t_span_polar = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        sol_polar = odeint(du_dt, t0, t_span_polar)
        sol_polar = np.maximum(sol_polar, 0.001)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': 'polar'})
        axes = axes.flatten()

        labels = ['L₁','L₂','L₃','L₄','L₅','L₆','L₇','L₈','L₉','L₁₀','L₁₁','L₁₂','L₁₃','L₁₄','L₁₅']
        angles = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        angles_closed = np.append(angles, angles[0])
        norm_bounds_closed = np.append(norm_bounds, norm_bounds[0])

        for i, ax in enumerate(axes):
            if i < len(t_span_polar):
                sol_values = np.append(sol_polar[i, :], sol_polar[i, 0])
                ax.plot(angles_closed, sol_values, linewidth=2)
                ax.fill(angles_closed, sol_values, alpha=0.25)
                ax.plot(angles_closed, norm_bounds_closed, linestyle='--', linewidth=1)
                ax.fill(angles_closed, norm_bounds_closed, alpha=0.1)
                ax.set_xticks(angles)
                ax.set_xticklabels(labels)
                ax.set_title(f'Время t = {t_span_polar[i]}', fontsize=12, pad=20)
            else:
                ax.axis('off')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        polar_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

    return plot_data, polar_plot_data


# ========= ROUTES =========

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_data = None
    polar_plot_data = None
    show_polar = False

    if request.method == 'POST':
        # Если в форме есть поле norm_bound_0 — значит запрос с формы полярных диаграмм,
        # но старый вариант тоже мог отправлять всё в index; учитываем оба случая.
        if any(k.startswith('norm_bound_') for k in request.form.keys()):
            # Построить только полярные
            plot_data, polar_plot_data = generate_plots_from_form(request.form, want_main_plot=False, want_polar=True)
            show_polar = True
        else:
            # Обычный расчёт динамики системы (главная кнопка)
            plot_data, polar_plot_data = generate_plots_from_form(request.form, want_main_plot=True, want_polar=False)
            show_polar = False

    # t_span для формы (как раньше)
    t_span = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    return render_template('index.html', v0=v0, c=c, f=f, t_span=t_span,
                           plot_data=plot_data, polar_plot_data=polar_plot_data, show_polar=show_polar)


@app.route('/polar_plot', methods=['POST'])
def polar_plot():
    """
    Дополнительный маршрут, на который указывает форма построения полярных диаграмм.
    Возвращает ту же index.html, но с polar_plot_data.
    """
    plot_data, polar_plot_data = generate_plots_from_form(request.form, want_main_plot=False, want_polar=True)
    t_span = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    return render_template('index.html', v0=v0, c=c, f=f, t_span=t_span,
                           plot_data=plot_data, polar_plot_data=polar_plot_data, show_polar=True)



if __name__ == '__main__':
    app.run(debug=True)