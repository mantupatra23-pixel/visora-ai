// lib/screens/dashboard_screen.dart
import 'dart:async';
import 'dart:math';
import 'dart:ui';

import 'package:blur/blur.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lottie/lottie.dart';

/// Visora Dashboard Screen
/// - Dynamic data placeholders (replace fetchData() with real API)
/// - Lightweight custom charts (line + pie + bar) implemented with CustomPainter
/// - Glassmorphism + animated gradient background
class DashboardScreen extends StatefulWidget {
  final Map<String, dynamic>? initialData;
  const DashboardScreen({Key? key, this.initialData}) : super(key: key);

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> with TickerProviderStateMixin {
  bool _loading = true;
  late AnimationController _bgController;
  late Animation<double> _bgAnim;

  // dynamic data model (defaults)
  Map<String, dynamic> data = {
    "userStats": {
      "totalVideos": 0,
      "renderMinutes": 0,
      "credits": 0,
      "creditsUsed": 0,
      "renderQueue": 0
    },
    "dailyStats": [
      {"day": "Mon", "renders": 0},
      {"day": "Tue", "renders": 0},
      {"day": "Wed", "renders": 0},
      {"day": "Thu", "renders": 0},
      {"day": "Fri", "renders": 0},
      {"day": "Sat", "renders": 0},
      {"day": "Sun", "renders": 0},
    ],
    "languageStats": [
      {"lang": "Hindi", "count": 0},
      {"lang": "English", "count": 0},
      {"lang": "Tamil", "count": 0}
    ],
    "qualityStats": [
      {"quality": "480p", "count": 0},
      {"quality": "720p", "count": 0},
      {"quality": "1080p", "count": 0},
      {"quality": "4K", "count": 0},
    ],
    "queueData": [],
  };

  @override
  void initState() {
    super.initState();
    _bgController = AnimationController(vsync: this, duration: const Duration(seconds: 14))..repeat(reverse: true);
    _bgAnim = Tween<double>(begin: -0.3, end: 0.3).animate(CurvedAnimation(parent: _bgController, curve: Curves.easeInOut));
    // if initialData passed (from main), use it
    if (widget.initialData != null) {
      data = widget.initialData!;
      _loading = false;
    } else {
      // simulate fetch
      _fetchData();
    }
  }

  @override
  void dispose() {
    _bgController.dispose();
    super.dispose();
  }

  Future<void> _fetchData() async {
    setState(() => _loading = true);
    // TODO: Replace the simulated delay with actual HTTP call to backend
    await Future.delayed(const Duration(milliseconds: 900));
    // Simulated realistic sample data
    final rnd = Random();
    data = {
      "userStats": {
        "totalVideos": 52,
        "renderMinutes": 310,
        "credits": 1400,
        "creditsUsed": 2600,
        "renderQueue": 2
      },
      "dailyStats": List.generate(7, (i) => {"day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i], "renders": rnd.nextInt(10)+1}),
      "languageStats": [
        {"lang": "Hindi", "count": 18},
        {"lang": "English", "count": 12},
        {"lang": "Tamil", "count": 6},
        {"lang": "Telugu", "count": 4},
        {"lang": "Marathi", "count": 3},
      ],
      "qualityStats": [
        {"quality": "480p", "count": 8},
        {"quality": "720p", "count": 15},
        {"quality": "1080p", "count": 20},
        {"quality": "4K", "count": 9},
      ],
      "queueData": [
        {"title": "AI Avatar V1", "status": "Running"},
        {"title": "Short Promo", "status": "Queued"},
        {"title": "Motivation - Hindi", "status": "Completed"},
      ],
    };
    setState(() => _loading = false);
  }

  Widget _animatedBackground() {
    return AnimatedBuilder(
      animation: _bgAnim,
      builder: (context, child) {
        final shift = _bgAnim.value;
        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment(-1.0 + shift, -0.5 - shift),
              end: Alignment(1.0 - shift, 0.5 + shift),
              colors: [
                Color.lerp(Colors.deepPurple.shade700, Colors.indigo.shade700, (shift + 0.5))!,
                Color.lerp(Colors.pink.shade700, Colors.orange.shade700, (1 - shift))!,
              ],
            ),
          ),
          child: CustomPaint(
            painter: _SoftTexturePainter(shift * 200),
            child: Container(),
          ),
        );
      },
    );
  }

  Widget _topBar() {
    final stats = data['userStats'] as Map<String, dynamic>;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 10),
        child: Row(
          children: [
            Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Dashboard', style: GoogleFonts.poppins(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w700)),
              const SizedBox(height: 4),
              Text('${stats['totalVideos']} videos • ${stats['renderMinutes']} mins', style: const TextStyle(color: Colors.white70, fontSize: 12)),
            ]),
            const Spacer(),
            IconButton(
              onPressed: _fetchData,
              icon: _loading ? const CircularProgressIndicator(color: Colors.white, strokeWidth: 2) : const Icon(Icons.refresh, color: Colors.white),
              tooltip: 'Refresh',
            ),
            const SizedBox(width: 6),
            CircleAvatar(backgroundColor: Colors.white12, child: const Icon(Icons.person, color: Colors.white)),
          ],
        ),
      ),
    );
  }

  Widget _overviewCards() {
    final stats = data['userStats'] as Map<String, dynamic>;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 8),
      child: Row(
        children: [
          _statCard('Videos', stats['totalVideos'].toString(), Icons.movie),
          const SizedBox(width: 8),
          _statCard('Minutes', stats['renderMinutes'].toString(), Icons.timer),
          const SizedBox(width: 8),
          _statCard('Credits', stats['credits'].toString(), Icons.credit_score),
          const SizedBox(width: 8),
          _statCard('Queue', stats['renderQueue'].toString(), Icons.queue),
        ],
      ),
    );
  }

  Widget _statCard(String title, String value, IconData icon) {
    return Expanded(
      child: Blur(
        blur: 6,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.03), borderRadius: BorderRadius.circular(12)),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: Colors.white10, borderRadius: BorderRadius.circular(10)),
                child: Icon(icon, color: Colors.white70),
              ),
              const SizedBox(width: 10),
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(value, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                const SizedBox(height: 4),
                Text(title, style: const TextStyle(color: Colors.white70, fontSize: 12)),
              ])
            ],
          ),
        ),
      ),
    );
  }

  Widget _chartsSection() {
    final daily = List<Map<String, dynamic>>.from(data['dailyStats'] as List<dynamic>);
    final lang = List<Map<String, dynamic>>.from(data['languageStats'] as List<dynamic>);
    final qual = List<Map<String, dynamic>>.from(data['qualityStats'] as List<dynamic>);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 12),
      child: Column(
        children: [
          // Line chart + Pie chart row
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                flex: 2,
                child: _glassCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('7-day Activity', style: TextStyle(color: Colors.white70)),
                      const SizedBox(height: 8),
                      SizedBox(height: 140, child: _LineChart(data: daily)),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                flex: 1,
                child: _glassCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Language Split', style: TextStyle(color: Colors.white70)),
                      const SizedBox(height: 8),
                      SizedBox(height: 140, child: _PieChart(data: lang)),
                      const SizedBox(height: 6),
                      Text('Top languages', style: const TextStyle(color: Colors.white54, fontSize: 12)),
                    ],
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _glassCard(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    const Text('Quality Usage', style: TextStyle(color: Colors.white70)),
                    const SizedBox(height: 8),
                    SizedBox(height: 110, child: _BarChart(data: qual)),
                  ]),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _glassCard({required Widget child}) {
    return Blur(
      blur: 6,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: Colors.white.withOpacity(0.03), borderRadius: BorderRadius.circular(12)),
        child: child,
      ),
    );
  }

  Widget _queueList() {
    final queue = List<Map<String, dynamic>>.from(data['queueData'] as List<dynamic>);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 12),
      child: Column(
        children: [
          _glassCard(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                const Text('Render Queue', style: TextStyle(color: Colors.white70)),
                const Spacer(),
                Text('${queue.length} jobs', style: const TextStyle(color: Colors.white54)),
              ]),
              const SizedBox(height: 8),
              if (queue.isEmpty)
                SizedBox(
                    height: 120,
                    child: Center(
                        child: Column(mainAxisSize: MainAxisSize.min, children: [
                      Lottie.asset('assets/lottie/loading.json', width: 80),
                      const SizedBox(height: 8),
                      const Text('No jobs in queue', style: TextStyle(color: Colors.white54))
                    ]))),
              ...queue.map((j) => ListTile(
                    tileColor: Colors.white10,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    title: Text(j['title'], style: const TextStyle(color: Colors.white)),
                    subtitle: Text(j['status'], style: const TextStyle(color: Colors.white70)),
                    trailing: j['status'] == 'Completed' ? IconButton(icon: const Icon(Icons.download), onPressed: () {}) : null,
                  )),
            ]),
          ),
        ],
      ),
    );
  }

  Widget _insights() {
    final stats = data['userStats'] as Map<String, dynamic>;
    final percent4k = ( (data['qualityStats'] as List).where((e) => e['quality'] == '4K').isNotEmpty
        ? (data['qualityStats'] as List).firstWhere((e) => e['quality'] == '4K')['count']
        : 0);
    final totalQual = (data['qualityStats'] as List).fold<int>(0, (p, e) => p + (e['count'] as int));
    final pct = totalQual == 0 ? 0 : ((percent4k as int) / totalQual * 100).round();
    final tips = <String>[];
    if (pct < 10) tips.add('Try publishing a 4K highlight for platform visibility.');
    if (stats['renderMinutes'] > 300) tips.add('You are a heavy user — consider premium for fast queue.');
    if (tips.isEmpty) tips.add('All systems normal — keep creating!');

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 12),
      child: _glassCard(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('System Insights', style: TextStyle(color: Colors.white70)),
          const SizedBox(height: 8),
          ...tips.map((t) => ListTile(leading: const Icon(Icons.lightbulb_outline, color: Colors.amber), title: Text(t, style: const TextStyle(color: Colors.white)))),
        ]),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(children: [
        _animatedBackground(),
        SafeArea(
          child: _loading
              ? Center(
                  child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                    Lottie.asset('assets/lottie/loading.json', width: 140),
                    const SizedBox(height: 12),
                    const Text('Fetching dashboard...', style: TextStyle(color: Colors.white70)),
                  ]),
                )
              : RefreshIndicator(
                  onRefresh: _fetchData,
                  child: SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.only(bottom: 80),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _topBar(),
                        const SizedBox(height: 8),
                        _overviewCards(),
                        _chartsSection(),
                        _queueList(),
                        _insights(),
                        const SizedBox(height: 40),
                      ],
                    ),
                  ),
                ),
        ),
      ]),
    );
  }
}

/// ----------------- Custom painters -----------------

class _LineChart extends StatelessWidget {
  final List<Map<String, dynamic>> data;
  const _LineChart({required this.data});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _LineChartPainter(data),
      child: Container(),
    );
  }
}

class _LineChartPainter extends CustomPainter {
  final List<Map<String, dynamic>> data;
  _LineChartPainter(this.data);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..style = PaintingStyle.stroke..strokeWidth = 2..shader = ui.Gradient.linear(Offset(0,0), Offset(size.width,0), [Colors.cyanAccent, Colors.blueAccent]);
    final fill = Paint()..style = PaintingStyle.fill..color = Colors.white.withOpacity(0.04);

    final maxVal = data.map((e) => e['renders'] as int).fold<int>(0, (p, n) => max(p, n));
    final spacing = size.width / (data.length - 1);
    final points = <Offset>[];
    for (int i = 0; i < data.length; i++) {
      final val = (data[i]['renders'] as int).toDouble();
      final dx = spacing * i;
      final dy = size.height - (maxVal == 0 ? 0 : (val / maxVal * size.height));
      points.add(Offset(dx, dy));
    }

    // fill path
    if (points.isNotEmpty) {
      final path = Path()..moveTo(points.first.dx, size.height)..lineTo(points.first.dx, points.first.dy);
      for (int i = 1; i < points.length; i++) path.lineTo(points[i].dx, points[i].dy);
      path.lineTo(points.last.dx, size.height);
      path.close();
      canvas.drawPath(path, fill);
      // line
      final linePath = Path()..moveTo(points.first.dx, points.first.dy);
      for (int i = 1; i < points.length; i++) linePath.lineTo(points[i].dx, points[i].dy);
      canvas.drawPath(linePath, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _LineChartPainter oldDelegate) => oldDelegate.data != data;
}

class _PieChart extends StatelessWidget {
  final List<Map<String, dynamic>> data;
  const _PieChart({required this.data});
  @override
  Widget build(BuildContext context) => CustomPaint(painter: _PieChartPainter(data));
}

class _PieChartPainter extends CustomPainter {
  final List<Map<String, dynamic>> data;
  _PieChartPainter(this.data);

  @override
  void paint(Canvas canvas, Size size) {
    final total = data.fold<int>(0, (p, e) => p + (e['count'] as int));
    final center = Offset(size.width / 2, size.height / 2);
    final radius = min(size.width, size.height) * 0.4;
    var start = -pi / 2.0;
    final paint = Paint()..style = PaintingStyle.fill;
    final colors = [Colors.pinkAccent, Colors.deepPurpleAccent, Colors.cyanAccent, Colors.orangeAccent, Colors.greenAccent];
    for (int i = 0; i < data.length; i++) {
      final slice = (total == 0) ? 0.0 : (data[i]['count'] as int) / total;
      final sweep = slice * 2 * pi;
      paint.color = colors[i % colors.length].withOpacity(0.9);
      canvas.drawArc(Rect.fromCircle(center: center, radius: radius), start, sweep, true, paint);
      start += sweep;
    }
    // center hole
    final hole = Paint()..color = Colors.black.withOpacity(0.3);
    canvas.drawCircle(center, radius * 0.45, hole);
  }

  @override
  bool shouldRepaint(covariant _PieChartPainter oldDelegate) => oldDelegate.data != data;
}

class _BarChart extends StatelessWidget {
  final List<Map<String, dynamic>> data;
  const _BarChart({required this.data});
  @override
  Widget build(BuildContext context) => CustomPaint(painter: _BarChartPainter(data));
}

class _BarChartPainter extends CustomPainter {
  final List<Map<String, dynamic>> data;
  _BarChartPainter(this.data);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..style = PaintingStyle.fill;
    final maxVal = data.fold<int>(0, (p, e) => max(p, (e['count'] as int)));
    final barWidth = size.width / (data.length * 2);
    for (int i = 0; i < data.length; i++) {
      final x = (i * 2 + 0.5) * barWidth;
      final val = (data[i]['count'] as int).toDouble();
      final h = maxVal == 0 ? 0 : (val / maxVal * size.height);
      paint.color = Colors.primaries[i % Colors.primaries.length].withOpacity(0.8);
      final rect = Rect.fromLTWH(x, size.height - h, barWidth, h);
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(6)), paint);
    }
  }

  @override
  bool shouldRepaint(covariant _BarChartPainter oldDelegate) => oldDelegate.data != data;
}

/// subtle textured painter
class _SoftTexturePainter extends CustomPainter {
  final double shift;
  _SoftTexturePainter(this.shift);
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..style = PaintingStyle.fill;
    for (int i = 0; i < 10; i++) {
      paint.color = Colors.white.withOpacity(0.01 + (i % 3) * 0.01);
      final x = (i * 123.7 + shift) % (size.width + 120) - 60;
      final y = (i * 77.3 + shift * 0.3) % (size.height + 100) - 50;
      canvas.drawCircle(Offset(x, y), 30 + (i % 4) * 8.0, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _SoftTexturePainter oldDelegate) => oldDelegate.shift != shift;
}
