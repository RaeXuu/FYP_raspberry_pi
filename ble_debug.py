"""
BLE 蓝牙调试工具
用途：
  1. 扫描并显示 ESP32 设备信息
  2. 连接后列出所有 GATT 服务与特征（查看蓝牙模式/配置）
  3. 订阅目标特征，实时显示接收到的原始数据
"""

import asyncio
import time
from bleak import BleakClient, BleakScanner

ESP32_MAC = "80:F1:B2:ED:B4:12"
TARGET_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# ==========================================
# 模式一：扫描并显示设备详情
# ==========================================
async def scan_devices():
    print("=" * 55)
    print("  [扫描模式] 正在搜索附近 BLE 设备（10 秒）")
    print("=" * 55)

    devices = await BleakScanner.discover(timeout=10.0, return_adv=True)

    print(f"共发现 {len(devices)} 个设备:\n")
    for addr, (device, adv) in devices.items():
        marker = "  >>>" if addr.upper() == ESP32_MAC.upper() else "     "
        name = device.name or "(无名称)"
        rssi = adv.rssi if adv.rssi is not None else "N/A"
        print(f"{marker} [{addr}]  {name:<20}  RSSI: {rssi} dBm")

    print()
    target = devices.get(ESP32_MAC.upper()) or devices.get(ESP32_MAC.lower())
    if target:
        device, adv = target
        print(f"✅ 目标 ESP32 已找到: {device.name} ({ESP32_MAC})")
        if adv.service_uuids:
            print(f"   广播服务 UUID:")
            for uuid in adv.service_uuids:
                print(f"     - {uuid}")
    else:
        print(f"❌ 未找到目标 ESP32 ({ESP32_MAC})，请确认设备已开机并在附近")


# ==========================================
# 模式二：连接并列出所有 GATT 服务/特征
# ==========================================
async def inspect_services():
    print("=" * 55)
    print(f"  [服务检查] 正在连接 {ESP32_MAC}")
    print("=" * 55)

    async with BleakClient(ESP32_MAC) as client:
        print(f"✅ 已连接！MTU 大小: {client.mtu_size} 字节\n")
        print("GATT 服务与特征列表:")
        print("-" * 55)

        for service in client.services:
            print(f"\n[服务] {service.uuid}")
            print(f"       描述: {service.description}")
            for char in service.characteristics:
                props = ", ".join(char.properties)
                target_mark = "  ← 目标特征" if char.uuid == TARGET_UUID else ""
                print(f"  [特征] {char.uuid}{target_mark}")
                print(f"         属性: {props}")
                print(f"         描述: {char.description}")

                # 如果可读，尝试读取当前值
                if "read" in char.properties:
                    try:
                        val = await client.read_gatt_char(char.uuid)
                        print(f"         当前值: {val.hex()} ({len(val)} 字节)")
                    except Exception as e:
                        print(f"         读取失败: {e}")

        print("\n" + "-" * 55)
        print("检查完毕。")


# ==========================================
# 模式三：订阅目标特征，实时显示原始数据
# ==========================================
async def monitor_data(duration: int = 30):
    print("=" * 55)
    print(f"  [数据监控] 连接中，监控时长 {duration} 秒")
    print(f"  目标 UUID: {TARGET_UUID}")
    print("=" * 55)

    stats = {"count": 0, "total_bytes": 0, "start": None}

    def handler(sender, data: bytearray):
        if stats["start"] is None:
            stats["start"] = time.time()

        stats["count"] += 1
        stats["total_bytes"] += len(data)
        elapsed = time.time() - stats["start"] if stats["start"] else 0

        # 显示前 16 字节的 hex
        hex_preview = data[:16].hex(" ") + ("..." if len(data) > 16 else "")
        # 尝试显示可打印 ASCII
        ascii_preview = "".join(chr(b) if 32 <= b < 127 else "." for b in data[:16])

        print(f"  #{stats['count']:04d} | {len(data):4d}B | "
              f"hex: {hex_preview:<50} | ascii: {ascii_preview}")

    try:
        async with BleakClient(ESP32_MAC) as client:
            await client._backend._acquire_mtu()
            print(f"✅ 已连接！MTU: {client.mtu_size} 字节")

            # 检查目标特征是否存在
            uuids = [c.uuid for s in client.services for c in s.characteristics]
            if TARGET_UUID not in uuids:
                print(f"❌ 目标特征 {TARGET_UUID} 不存在！")
                print("   可用特征:", uuids)
                return

            await client.start_notify(TARGET_UUID, handler)
            print(f"📥 开始接收数据，等待 {duration} 秒...\n")
            print(f"  {'包号':<6} {'大小':>5} | {'HEX (前16字节)':<50} | ASCII")
            print("  " + "-" * 80)

            await asyncio.sleep(duration)
            await client.stop_notify(TARGET_UUID)
    except EOFError:
        pass

    print("\n" + "=" * 55)
    print("  统计结果:")
    if stats["count"] > 0:
        elapsed = time.time() - stats["start"]
        rate = stats["total_bytes"] / elapsed / 1024
        print(f"  接收包数:    {stats['count']}")
        print(f"  总数据量:    {stats['total_bytes']} 字节 ({stats['total_bytes']/1024:.2f} KB)")
        print(f"  平均速率:    {rate:.2f} KB/s")
        print(f"  平均包大小:  {stats['total_bytes'] / stats['count']:.1f} 字节")
    else:
        print("  未收到任何数据。请确认 ESP32 正在发送。")
    print("=" * 55)


# ==========================================
# 主菜单
# ==========================================
async def main():
    print("\nBLE 调试工具")
    print("1. 扫描设备")
    print("2. 检查 GATT 服务与特征")
    print("3. 监控原始数据（30秒）")
    print("4. 监控原始数据（自定义时长）")
    print()

    choice = input("请选择 (1-4): ").strip()

    if choice == "1":
        await scan_devices()
    elif choice == "2":
        await inspect_services()
    elif choice == "3":
        await monitor_data(30)
    elif choice == "4":
        sec = input("监控时长（秒）: ").strip()
        await monitor_data(int(sec))
    else:
        print("无效选项")


if __name__ == "__main__":
    asyncio.run(main())
