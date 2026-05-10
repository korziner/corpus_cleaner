use bloomfilter::Bloom;
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, ValueEnum, Debug)]
enum InputFormat {
    Text,
    Json,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Утилита для фильтрации плохо распознанных (OCR) текстов на основе словаря.",
    long_about = "
Высокопроизводительная утилита на Rust для очистки текстовых корпусов перед обучением LLM.
Использует фильтры Блума (Bloom Filter) для мгновенного поиска по словарю (~1.5 млн слов)
и Rayon для многопоточной обработки.

Алгоритм:
Текст разбивается на слова. По тексту скользит 'плавающее окно' заданного размера.
Если в каком-либо окне количество валидных (словарных) слов падает ниже порога,
вся строка (или JSON-объект) отбрасывается как 'забагованная'.

Режимы работы:
1. Ручной: задать --window и --min-valid.
2. Автопоиск: указать --auto-search, программа переберёт окна и найдёт самый строгий
   порог min_valid, сохраняющий заданный процент строк (--target-keep-percent).

Алгориѳмъ ищетъ наистрожайшую сочетаемость параметровъ (т. е. наибольшiй min_valid при заданномъ window), коя ещё удержываетъ не менѣе заданнаго числа строкъ. Ежели же для нѣкоего window обрѣтается нѣсколько значенiй min_valid, исполняющихъ сiе условiе, избирается самое высокое. При равенствѣ min_valid предпочтенiе отдаётся меньшему window.

Почему такъ?
Чѣмъ выше min_valid, тѣмъ жёстче бракуется «испорченный» текстъ. Цѣлевой процентъ сохранённыхъ строкъ есть лишь нижнiй предѣлъ; превышенiе онаго не вредитъ – качество очистки остаётся приоритетомъ. Иное дѣло, коль скоро нѣтъ ни единой комбинацiи, удосужной удержать хотя бы заданное число строкъ. Въ такомъ случаѣ алгоритмъ возвѣщаетъ о неудачѣ, дабы пользователь властенъ былъ умалить либо target-keep-percent, либо размахъ поиска (min-search-window/max-search-window).

Слѣдовательно, полученное рѣшенiе – это самая «чистая» выборка, каковую только можно извлечь при соблюденiи минимальнаго объёма сохранённыхъ данныхъ. Пожертвовать полнотой ради точности попаданiя въ процентъ было бы противно здравому смыслу очистки корпуса передъ обученiемъ LLM.

time zstdcat *.jsonl.zst|gemini31-pro-preview_corpus_cleaner --dict ./prereform_words_no-bugs.txt --format json   --auto-search --target-keep-percent 10     
Загрузка словаря из ./prereform_words_no-bugs.txt...
Словарь загружен: 1503789 слов за 1.04s. Фильтр Блума готов.
Чтение входных данных...
Прочитано 120301 строк за 27.38s
Предвычисление флагов слов...
Предвычисление завершено для 74682 строк за 101.93s
--- АВТОПОИСК: целевое количество строк = 12030 (10.0%) ---
Окно 3: ищем best min_valid... нет подходящего min_valid
Окно 4: ищем best min_valid... нет подходящего min_valid
Окно 5: ищем best min_valid... нет подходящего min_valid
Окно 6: ищем best min_valid... нашли min_valid=1, сохраняется 22639 строк (18.8%) за 531.00ms
Окно 7: ищем best min_valid... нашли min_valid=1, сохраняется 35834 строк (29.8%) за 768.23ms
Окно 8: ищем best min_valid... нашли min_valid=2, сохраняется 19470 строк (16.2%) за 356.19ms
Окно 9: ищем best min_valid... нашли min_valid=2, сохраняется 29844 строк (24.8%) за 660.94ms
Окно 10: ищем best min_valid... нашли min_valid=3, сохраняется 19790 строк (16.5%) за 829.35ms
Окно 11: ищем best min_valid... нашли min_valid=4, сохраняется 13570 строк (11.3%) за 811.45ms
Окно 12: ищем best min_valid... нашли min_valid=4, сохраняется 20874 строк (17.4%) за 1.18s
Окно 13: ищем best min_valid... нашли min_valid=5, сохраняется 15632 строк (13.0%) за 1.05s
Окно 14: ищем best min_valid... нашли min_valid=6, сохраняется 12066 строк (10.0%) за 1.29s
Окно 15: ищем best min_valid... нашли min_valid=6, сохраняется 17693 строк (14.7%) за 1.49s
Окно 16: ищем best min_valid... нашли min_valid=7, сохраняется 14323 строк (11.9%) за 1.12s
Окно 17: ищем best min_valid... нашли min_valid=7, сохраняется 19757 строк (16.4%) за 1.74s
Окно 18: ищем best min_valid... нашли min_valid=8, сохраняется 16467 строк (13.7%) за 2.06s
Окно 19: ищем best min_valid... нашли min_valid=9, сохраняется 13885 строк (11.5%) за 2.00s
Окно 20: ищем best min_valid... нашли min_valid=9, сохраняется 18704 строк (15.5%) за 2.38s

=== ЛУЧШАЯ КОМБИНАЦИЯ: window=19, min_valid=9 (сохраняет 13885 строк = 11.5%) ===
Результат сохранён в clean.window19--min-valid9.13885=11.5%.txt

real    2m43,450s
user    8m

"
)]
struct Args {
    /// Путь к файлу словаря (одно слово на строку)
    #[arg(short, long)]
    dict: PathBuf,

    /// Входной файл (если не указан, читается из stdin)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Формат входных данных
    #[arg(short, long, value_enum, default_value_t = InputFormat::Text)]
    format: InputFormat,

    /// Имя поля с текстом, если формат json
    #[arg(long, default_value = "text")]
    json_field: String,

    /// Размер плавающего окна в словах (для ручного режима)
    #[arg(short, long, default_value_t = 7)]
    window: usize,

    /// Минимальное количество валидных слов в окне (для ручного режима)
    #[arg(short, long, default_value_t = 4)]
    min_valid: usize,

    /// Допустимая вероятность ложноположительного срабатывания фильтра Блума
    #[arg(long, default_value_t = 0.001)]
    fp_rate: f64,

    /// Включить автоматический подбор параметров (перебирает окна, двоичный поиск по min_valid)
    #[arg(long)]
    auto_search: bool,

    /// Целевой процент сохраняемых строк (0-100) для автопоиска
    #[arg(long, default_value_t = 50.0)]
    target_keep_percent: f64,

    /// Минимальный размер окна при автопоиске
    #[arg(long, default_value_t = 3)]
    min_search_window: usize,

    /// Максимальный размер окна при автопоиске
    #[arg(long, default_value_t = 20)]
    max_search_window: usize,

    /// Директория для сохранения результатов (только при автопоиске или явном --output-dir)
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,

    /// Шаблон имени выходного файла. Доступны подстановки:
    /// {W} - окно, {M} - min-valid, {K} - кол-во сохранённых строк, {P} - процент.
    /// По умолчанию: "clean.window{W}--min-valid{M}.{K}={P:.1}%.txt"
    #[arg(long, default_value = "clean.window{W}--min-valid{M}.{K}={P}%.txt")]
    output_pattern: String,

}

// ------------------------------------------------------------
// Структура для предвычисленных флагов валидности слов
// ------------------------------------------------------------
struct LineData {
    original: String,
    valid_flags: Vec<bool>,
}

// ------------------------------------------------------------
// Загрузка словаря в фильтр Блума
// ------------------------------------------------------------
fn build_bloom_filter(dict_path: &PathBuf, fp_rate: f64) -> Bloom<String> {
    eprintln!("Загрузка словаря из {:?}...", dict_path);
    let start = Instant::now();

    let file = File::open(dict_path).expect("Не удалось открыть файл словаря");
    let reader = BufReader::new(file);

    let words: Vec<String> = reader
        .lines()
        .filter_map(Result::ok)
        .map(|w| w.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();

    let expected_items = words.len();
    let mut bloom = Bloom::new_for_fp_rate(expected_items, fp_rate)
        .expect("Не удалось создать фильтр Блума с заданной вероятностью");
    for word in &words {
        bloom.set(word);
    }

    eprintln!(
        "Словарь загружен: {} слов за {:.2?}. Фильтр Блума готов.",
        words.len(),
        start.elapsed()
    );
    bloom
}

// ------------------------------------------------------------
// Токенизация: извлекаем только буквенные последовательности
// ------------------------------------------------------------
fn extract_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ------------------------------------------------------------
// Извлечение текста из строки в зависимости от формата (возвращает владеющую строку)
// ------------------------------------------------------------
fn extract_text(line: &str, format: &InputFormat, json_field: &str) -> Option<String> {
    match format {
        InputFormat::Text => Some(line.to_string()),
        InputFormat::Json => {
            let json: Value = serde_json::from_str(line).ok()?;
            json.get(json_field)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        }
    }
}

// ------------------------------------------------------------
// Проверка качества текста по предвычисленным флагам слов
// ------------------------------------------------------------
fn is_clean_from_flags(flags: &[bool], window_size: usize, min_valid: usize) -> bool {
    if flags.is_empty() {
        return false;
    }
    if flags.len() < window_size {
        let required =
            (flags.len() as f64 * (min_valid as f64 / window_size as f64)).ceil() as usize;
        return flags.iter().filter(|&&v| v).count() >= required;
    }
    for window in flags.windows(window_size) {
        let valid_count = window.iter().filter(|&&v| v).count();
        if valid_count < min_valid {
            return false;
        }
    }
    true
}

// ------------------------------------------------------------
// Предвычисление флагов для всех строк (один проход)
// ------------------------------------------------------------
fn precompute_flags(
    lines: &[String],
    bloom: &Bloom<String>,
    format: &InputFormat,
    json_field: &str,
) -> Vec<LineData> {
    lines
        .par_iter()
        .filter_map(|line| {
            let text = extract_text(line, format, json_field)?;
            let words = extract_words(&text);
            let valid_flags: Vec<bool> = words.iter().map(|w| bloom.check(w)).collect();
            Some(LineData {
                original: line.clone(),
                valid_flags,
            })
        })
        .collect()
}

// ------------------------------------------------------------
// Оценка количества строк, прошедших фильтр для заданных параметров
// ------------------------------------------------------------
fn evaluate_params(data: &[LineData], window: usize, min_valid: usize) -> usize {
    data.par_iter()
        .filter(|line| is_clean_from_flags(&line.valid_flags, window, min_valid))
        .count()
}

// ------------------------------------------------------------
// Фильтрация строк по заданным параметрам (возвращает оригинальные строки)
// ------------------------------------------------------------
fn filter_by_params(data: &[LineData], window: usize, min_valid: usize) -> Vec<String> {
    data.par_iter()
        .filter(|line| is_clean_from_flags(&line.valid_flags, window, min_valid))
        .map(|line| line.original.clone())
        .collect()
}

// ------------------------------------------------------------
// Генерация имени файла по шаблону
// ------------------------------------------------------------
fn generate_filename(pattern: &str, window: usize, min_valid: usize, kept: usize, total: usize) -> String {
    let percent = if total == 0 { 0.0 } else { (kept as f64 / total as f64) * 100.0 };
    pattern
        .replace("{W}", &window.to_string())
        .replace("{M}", &min_valid.to_string())
        .replace("{K}", &kept.to_string())
        .replace("{P}", &format!("{:.1}", percent))
}


// ------------------------------------------------------------
// Двоичный поиск наилучшего min_valid для фиксированного окна
// ------------------------------------------------------------
fn best_min_valid_for_window(
    data: &[LineData],
    window: usize,
    target_keep: usize,
) -> Option<usize> {
    if window == 0 {
        return None;
    }
    let (mut lo, mut hi) = (1, window);
    let mut best = None;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let kept = evaluate_params(data, window, mid);
        if kept >= target_keep {
            best = Some(mid);
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    best
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
fn main() {
    let args = Args::parse();

    if !args.auto_search && args.min_valid > args.window {
        eprintln!("Ошибка: min-valid не может быть больше размера окна (window).");
        std::process::exit(1);
    }

    // 1. Загружаем словарь
    let bloom = build_bloom_filter(&args.dict, args.fp_rate);

    // 2. Читаем все строки ввода
    let stdin = io::stdin();
    let reader: Box<dyn BufRead> = match &args.input {
        Some(path) => Box::new(BufReader::new(File::open(path).expect("Не удалось открыть входной файл"))),
        None => Box::new(stdin.lock()),
    };

    eprintln!("Чтение входных данных...");
    let start_read = Instant::now();
    let all_lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();
    let total_lines = all_lines.len();
    eprintln!("Прочитано {} строк за {:.2?}", total_lines, start_read.elapsed());

    if total_lines == 0 {
        eprintln!("Нет данных для обработки.");
        return;
    }

    // 3. Предвычисление флагов
    eprintln!("Предвычисление флагов слов...");
    let start_pre = Instant::now();
    let data = precompute_flags(&all_lines, &bloom, &args.format, &args.json_field);
    eprintln!(
        "Предвычисление завершено для {} строк за {:.2?}",
        data.len(),
        start_pre.elapsed()
    );

    // 4. Режим автопоиска или ручной
    if args.auto_search {
        let target_keep = (total_lines as f64 * args.target_keep_percent / 100.0).round() as usize;
        eprintln!(
            "--- АВТОПОИСК: целевое количество строк = {} ({:.1}%) ---",
            target_keep, args.target_keep_percent
        );

        let mut best_combination = None;
        let mut best_min_valid = 0;
        let mut best_window = 0;

        for window in args.min_search_window..=args.max_search_window {
            eprint!("Окно {}: ищем best min_valid... ", window);
            let start_win = Instant::now();
            if let Some(minv) = best_min_valid_for_window(&data, window, target_keep) {
                let kept = evaluate_params(&data, window, minv);
                eprintln!(
                    "нашли min_valid={}, сохраняется {} строк ({:.1}%) за {:.2?}",
                    minv,
                    kept,
                    (kept as f64 / total_lines as f64) * 100.0,
                    start_win.elapsed()
                );
                if minv > best_min_valid || (minv == best_min_valid && window < best_window) {
                    best_min_valid = minv;
                    best_window = window;
                    best_combination = Some((window, minv));
                }
            } else {
                eprintln!("нет подходящего min_valid");
            }
        }

        if let Some((window, min_valid)) = best_combination {
            let kept_count = evaluate_params(&data, window, min_valid);
            let percent = (kept_count as f64 / total_lines as f64) * 100.0;
            eprintln!(
                "\n=== ЛУЧШАЯ КОМБИНАЦИЯ: window={}, min_valid={} (сохраняет {} строк = {:.1}%) ===",
                window, min_valid, kept_count, percent
            );

            let kept_lines = filter_by_params(&data, window, min_valid);
            let filename = generate_filename(
                &args.output_pattern,
                window,
                min_valid,
                kept_lines.len(),
                total_lines,
            );
            let out_path = args.output_dir.join(filename);
            let mut out_file = File::create(&out_path).expect("Не удалось создать выходной файл");
            for line in kept_lines {
                writeln!(out_file, "{}", line).expect("Ошибка записи");
            }
            eprintln!("Результат сохранён в {:?}", out_path);
        } else {
            eprintln!(
                "Не найдено комбинации, сохраняющей хотя бы {} строк ({}%).",
                target_keep, args.target_keep_percent
            );
        }
    } else {
        eprintln!(
            "--- РУЧНОЙ РЕЖИМ: window={}, min_valid={} ---",
            args.window, args.min_valid
        );
        let kept_lines = filter_by_params(&data, args.window, args.min_valid);
        let kept_count = kept_lines.len();
        let percent = (kept_count as f64 / total_lines as f64) * 100.0;
        eprintln!(
            "Сохранено: {} из {} ({:.1}%)",
            kept_count, total_lines, percent
        );

        // Если пользователь явно указал output_dir или изменил output_pattern, пишем в файл, иначе в stdout
        let use_file = args.output_dir != PathBuf::from(".") || args.output_pattern != "clean.window{W}--min-valid{M}.{K}={P:.1}%.txt";
        if use_file {
            let filename = generate_filename(
                &args.output_pattern,
                args.window,
                args.min_valid,
                kept_lines.len(),
                total_lines,
            );
            let out_path = args.output_dir.join(filename);
            let mut out_file = File::create(&out_path).expect("Не удалось создать выходной файл");
            for line in kept_lines {
                writeln!(out_file, "{}", line).expect("Ошибка записи");
            }
            eprintln!("Результат сохранён в {:?}", out_path);
        } else {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            for line in kept_lines {
                writeln!(handle, "{}", line).expect("Ошибка вывода");
            }
        }
    }
}
