import asyncio
import multiprocessing

from implementation import Pipeline


async def main():
    multiprocessing.freeze_support()

    print("\n" + "=" * 50)
    print("ЗАМЕР ВРЕМЕНИ: Pipeline")
    print("=" * 50)

    pipeline = Pipeline(limit=3)

    start_time = asyncio.get_event_loop().time()

    await pipeline.run()

    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time

    print(f"\nPipeline выполнился за: {elapsed:.2f} секунд")

if __name__ == "__main__":
    asyncio.run(main())