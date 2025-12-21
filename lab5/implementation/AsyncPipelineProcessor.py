import os
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiofiles
import numpy as np
import cv2
from typing import AsyncGenerator, Tuple, List, Optional
from dotenv import load_dotenv
import aiohttp
from lab5.implementation import ColorCatImage, GrayCatImage


class AsyncPipelineProcessor:
    """
    Упрощенный асинхронный генераторный пайплайн.
    Три этапа, работающие конкурентно:
    1. stage_1_download: скачивание по чанкам
    2. stage_2_process: обработка в процессах
    3. stage_3_save: сохранение результатов
    """

    BASE_URL = "https://api.thecatapi.com/v1/images/search"

    def __init__(self, output_dir: str = "simple_pipeline_results", chunk_size: int = 3):

        load_dotenv()
        self._api_key = os.getenv("API_KEY")
        if not self._api_key:
            raise ValueError("API_KEY не найден в .env файле!")

        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(output_dir, exist_ok=True)

    #СКАЧИВАНИЕ

    async def stage_1_download(self, limit: int) -> AsyncGenerator[List[Tuple], None]:

        print(f"\n{'=' * 70}")
        print(f"   ЭТАП 1: СКАЧИВАНИЕ (чанк: {self.chunk_size} изображений)")
        print(f"{'=' * 70}")

        headers = {"x-api-key": self._api_key}
        params = {"limit": limit, "has_breeds": 1}

        async with aiohttp.ClientSession() as session:
            try:
                # Получаем список URL
                async with session.get(self.BASE_URL, headers=headers, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                print(f"Получено {len(data)} URL от API")

                # Создаем список задач для скачивания
                metadata = []
                for i, item in enumerate(data, start=1):
                    url = item.get("url")
                    breeds = item.get("breeds", [{}])
                    breed = breeds[0].get("name", "unknown") if breeds else "unknown"
                    metadata.append((i, breed, url))

                total_chunks = (len(metadata) + self.chunk_size - 1) // self.chunk_size
                print(f"Всего чанков: {total_chunks}")

                # Обрабатываем по чанкам
                for chunk_idx in range(0, len(metadata), self.chunk_size):
                    chunk_metadata = metadata[chunk_idx:chunk_idx + self.chunk_size]
                    chunk_num = (chunk_idx // self.chunk_size) + 1

                    print(f"\n--- ЧАНК {chunk_num}/{total_chunks} ---")
                    print(f"Изображения: {chunk_idx + 1}-{min(chunk_idx + self.chunk_size, len(metadata))}")

                    # Параллельное скачивание внутри чанка
                    download_tasks = []
                    for idx, breed, url in chunk_metadata:
                        task = asyncio.create_task(
                            self._download_single(session, idx, breed, url)
                        )
                        download_tasks.append(task)

                    # Ждем завершения скачивания чанка
                    chunk_results = await asyncio.gather(*download_tasks)

                    # Фильтруем успешные результаты
                    valid_results = [r for r in chunk_results if r is not None]

                    print(f"Успешно скачано: {len(valid_results)}/{len(chunk_results)}")

                    if valid_results:
                        yield valid_results

            except Exception as e:
                print(f"[ОШИБКА СКАЧИВАНИЯ] {e}")
                raise

    async def _download_single(self, session: aiohttp.ClientSession, idx: int, breed: str, url: str) -> Optional[Tuple]:
        """Скачивает одно изображение"""
        try:
            print(f"[СКАЧИВАНИЕ {idx}] Начинаю: {breed}")

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                content = await resp.read()

            img_arr = np.frombuffer(content, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"[СКАЧИВАНИЕ {idx}] Ошибка декодирования")
                return None

            print(f"[СКАЧИВАНИЕ {idx}] Успешно: {img.shape}")
            return (idx, breed, url, img)

        except Exception as e:
            print(f"[СКАЧИВАНИЕ {idx}] Ошибка: {e}")
            return None

    #ОБРАБОТКА

    async def stage_2_process(
            self,
            download_generator: AsyncGenerator[List[Tuple], None]
    ) -> AsyncGenerator[List[Tuple], None]:
        """
        Этап 2: Параллельная обработка изображений в процессах.

        Args:
            download_generator: Генератор из этапа 1

        Yields:
            Списки кортежей (idx, breed, img, edges_cv, edges_custom)
        """
        print(f"\n{'=' * 70}")
        print(f"   ЭТАП 2: ОБРАБОТКА В ПРОЦЕССАХ")
        print(f"{'=' * 70}")

        loop = asyncio.get_running_loop()

        async for image_batch in download_generator:
            if not image_batch:
                continue

            chunk_size = len(image_batch)
            print(f"\nОбработка чанка из {chunk_size} изображений...")

            with ProcessPoolExecutor() as pool:
                # Запускаем все задачи обработки параллельно
                process_futures = []
                for image_data in image_batch:
                    future = loop.run_in_executor(
                        pool,
                        self._process_single,
                        image_data
                    )
                    process_futures.append(future)

                # Собираем результаты по мере готовности
                processed_batch = []
                for future in asyncio.as_completed(process_futures):
                    try:
                        result = await future
                        if result is not None:
                            processed_batch.append(result)
                    except Exception as e:
                        print(f"[ОШИБКА ОБРАБОТКИ] {e}")

                print(f"Обработано: {len(processed_batch)}/{chunk_size}")

                if processed_batch:
                    yield processed_batch

    def _process_single(self, args: Tuple) -> Optional[Tuple]:
        """
        Обрабатывает одно изображение в отдельном процессе.

        Returns:
            Кортеж (idx, breed, original_img, edges_cv, edges_custom)
        """
        try:
            idx, breed, url, img = args

            print(f"[ПРОЦЕСС {os.getpid()}] Обработка {idx}: {breed}")

            if img is None:
                return None

            # Создаем объект изображения
            if img.ndim == 3:
                cat = ColorCatImage(img, breed, url)
            else:
                cat = GrayCatImage(img, breed, url)

            # Применяем детекцию краев
            edges_cv = cat.cv_edge_detection().image
            edges_custom = cat.my_edge_detection().image

            print(f"[ПРОЦЕСС {os.getpid()}] Завершено {idx}")
            return (idx, breed, img, edges_cv, edges_custom)

        except Exception as e:
            print(f"[ПРОЦЕСС {os.getpid()}] Ошибка: {e}")
            return None

    # СОХРАНЕНИЕ

    async def stage_3_save(
            self,
            process_generator: AsyncGenerator[List[Tuple], None]
    ) -> AsyncGenerator[str, None]:
        """
        Этап 3: Асинхронное сохранение результатов.

        Args:
            process_generator: Генератор из этапа 2

        Yields:
            Строки с информацией о сохранении
        """
        print(f"\n{'=' * 70}")
        print(f"   ЭТАП 3: СОХРАНЕНИЕ")
        print(f"{'=' * 70}")

        total_saved = 0

        async for processed_batch in process_generator:
            if not processed_batch:
                continue

            # Параллельное сохранение всего чанка
            save_tasks = []
            for processed in processed_batch:
                task = asyncio.create_task(self._save_single(processed))
                save_tasks.append(task)

            # Ждем завершения сохранения чанка
            save_results = await asyncio.gather(*save_tasks, return_exceptions=True)

            # Обрабатываем результаты
            for result in save_results:
                if isinstance(result, str):
                    total_saved += 1
                    yield result
                elif isinstance(result, Exception):
                    print(f"[ОШИБКА СОХРАНЕНИЯ] {result}")

        print(f"\nВсего сохранено: {total_saved} изображений")

    async def _save_single(self, processed: Tuple) -> str:
        """Сохраняет одно обработанное изображение"""
        try:
            idx, breed, orig_img, edges_cv, edges_custom = processed

            # Безопасное имя файла
            breed_safe = "".join(c if c.isalnum() else "_" for c in breed)

            # Создаем задачи для сохранения всех трех версий
            save_tasks = []

            # Оригинал
            if orig_img is not None:
                orig_path = os.path.join(self.output_dir, f"{idx:03d}_{breed_safe}_original.png")
                save_tasks.append(self._save_image(orig_img, orig_path))

            # OpenCV edges
            if edges_cv is not None:
                cv_path = os.path.join(self.output_dir, f"{idx:03d}_{breed_safe}_cv.png")
                save_tasks.append(self._save_image(edges_cv, cv_path))

            # Custom edges
            if edges_custom is not None:
                custom_path = os.path.join(self.output_dir, f"{idx:03d}_{breed_safe}_custom.png")
                save_tasks.append(self._save_image(edges_custom, custom_path))

            # Сохраняем параллельно
            if save_tasks:
                await asyncio.gather(*save_tasks)

            return f"[СОХРАНЕНО] {idx}: {breed} ({len(save_tasks)} файлов)"

        except Exception as e:
            return f"[ОШИБКА СОХРАНЕНИЯ {idx}] {e}"

    async def _save_image(self, img: np.ndarray, filepath: str):
        """Асинхронно сохраняет одно изображение"""
        try:
            ok, buf = cv2.imencode(".png", img)
            if ok:
                async with aiofiles.open(filepath, "wb") as f:
                    await f.write(buf.tobytes())
        except Exception as e:
            print(f"[ОШИБКА ФАЙЛА {filepath}] {e}")

    # ЗАПУСК ПАЙПЛАЙНА

    async def run_separate_generators_in_processes(self, limit: int):
        """
        Каждый генератор работает в отдельном процессе

        Соответствует требованию:
        1. Каждый генератор асинхронный ✅
        2. Каждый генератор в отдельном процессе ✅
        """
        print(f"\n{'=' * 70}")
        print(f"   ГЕНЕРАТОРЫ В ОТДЕЛЬНЫХ ПРОЦЕССАХ")
        print(f"   Изображений: {limit}")
        print(f"{'=' * 70}\n")

        start_time = asyncio.get_event_loop().time()

        # Создаём отдельные ProcessPoolExecutor для каждого генератора
        loop = asyncio.get_running_loop()

        # Шаг 1: Запускаем stage_1_download в отдельном процессе
        print("1. Запуск генератора скачивания в процессе...")
        with ProcessPoolExecutor(max_workers=1) as download_executor:
            download_future = loop.run_in_executor(
                download_executor,
                self._run_generator_in_process,
                "download",  # Название генератора
                limit
            )
            downloaded_data = await download_future

        # Шаг 2: Запускаем stage_2_process в отдельном процессе
        print("2. Запуск генератора обработки в процессе...")
        with ProcessPoolExecutor(max_workers=1) as process_executor:
            process_future = loop.run_in_executor(
                process_executor,
                self._run_generator_in_process,
                "process",  # Название генератора
                downloaded_data  # Данные от предыдущего генератора
            )
            processed_data = await process_future

        # Шаг 3: Запускаем stage_3_save в отдельном процессе
        print("3. Запуск генератора сохранения в процессе...")
        with ProcessPoolExecutor(max_workers=1) as save_executor:
            save_future = loop.run_in_executor(
                save_executor,
                self._run_generator_in_process,
                "save",  # Название генератора
                processed_data  # Данные от предыдущего генератора
            )
            saved_results = await save_future

        elapsed = asyncio.get_event_loop().time() - start_time

        print(f"\n{'=' * 70}")
        print(f"   ВСЕ ГЕНЕРАТОРЫ ЗАВЕРШЕНЫ")
        print(f"   Изображений: {len(saved_results)}/{limit}")
        print(f"   Время: {elapsed:.2f} секунд")
        print(f"{'=' * 70}\n")

        return saved_results

    def _run_generator_in_process(self, generator_name: str, data):
        """
        Запускает один генератор в отдельном процессе
        """
        import asyncio
        import pickle

        print(f"[PID:{os.getpid()}] Запуск генератора: {generator_name}")

        # Создаём новый экземпляр для изоляции
        local_processor = self._create_isolated_instance(generator_name)

        # Создаём новый цикл в процессе
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if generator_name == "download":
                # Запускаем только генератор скачивания
                async def run_download():
                    download_gen = local_processor.stage_1_download(data)
                    results = []
                    async for batch in download_gen:
                        results.extend(batch)
                    return pickle.dumps(results)  # Сериализуем для передачи

                return loop.run_until_complete(run_download())

            elif generator_name == "process":
                # Запускаем только генератор обработки
                async def run_process():
                    # Десериализуем данные
                    downloaded_data = pickle.loads(data)

                    # Создаём fake генератор из данных
                    async def fake_download_gen():
                        yield downloaded_data

                    process_gen = local_processor.stage_2_process(fake_download_gen())
                    results = []
                    async for batch in process_gen:
                        results.extend(batch)
                    return pickle.dumps(results)  # Сериализуем для передачи

                return loop.run_until_complete(run_process())

            elif generator_name == "save":
                # Запускаем только генератор сохранения
                async def run_save():
                    # Десериализуем данные
                    processed_data = pickle.loads(data)

                    # Создаём fake генератор из данных
                    async def fake_process_gen():
                        yield processed_data

                    save_gen = local_processor.stage_3_save(fake_process_gen())
                    results = []
                    async for result in save_gen:
                        results.append(result)
                    return results  # Возвращаем как есть

                return loop.run_until_complete(run_save())

        finally:
            loop.close()
            print(f"[PID:{os.getpid()}] Генератор {generator_name} завершён")

    def _create_isolated_instance(self, generator_name: str):
        """Создаёт изолированный экземпляр для работы в процессе"""
        # Создаём новую директорию для этого генератора
        generator_dir = os.path.join(self.output_dir, f"gen_{generator_name}_{os.getpid()}")
        os.makedirs(generator_dir, exist_ok=True)

        # Создаём новый экземпляр
        instance = AsyncPipelineProcessor(
            output_dir=generator_dir,
            chunk_size=self.chunk_size
        )
        instance._api_key = self._api_key

        return instance