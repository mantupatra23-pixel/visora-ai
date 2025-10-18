// lib/main.dart
import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:lottie/lottie.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:ui';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
  ));
  runApp(VisoraApp());
}

class VisoraApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Visora - AI Video Studio',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        textTheme: GoogleFonts.poppinsTextTheme(),
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: MainRouter(),
    );
  }
}

/// Router with Bottom Navigation
class MainRouter extends StatefulWidget {
  @override
  _MainRouterState createState() => _MainRouterState();
}

class _MainRouterState extends State<MainRouter> {
  int idx = 0;
  final pages = [HomeScreen(), TemplatesScreen(), DashboardScreen(), ProfileScreen()];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: pages[idx],
      bottomNavigationBar: NavigationBar(
        selectedIndex: idx,
        onDestinationSelected: (i) => setState(() => idx = i),
        destinations: [
          NavigationDestination(icon: Icon(Icons.home_filled), label: 'Home'),
          NavigationDestination(icon: Icon(Icons.layers_outlined), label: 'Templates'),
          NavigationDestination(icon: Icon(Icons.dashboard_customize), label: 'Dashboard'),
          NavigationDestination(icon: Icon(Icons.person_outline), label: 'Profile'),
        ],
      ),
    );
  }
}

/// ---------------- Home Screen (Polished with Lottie + Generate)
class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final TextEditingController _scriptCtrl = TextEditingController();
  bool _isLoading = false;
  String _jobId = '';
  String _previewUrl = '';
  late AnimationController _bgController;
  late Animation<double> _bgAnim;

  final String _backendUrl = 'https://example.com/generate-video'; // change

  @override
  void initState() {
    super.initState();
    _bgController = AnimationController(vsync: this, duration: Duration(seconds: 12))..repeat(reverse: true);
    _bgAnim = Tween<double>(begin: -0.3, end: 0.3).animate(CurvedAnimation(parent: _bgController, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _bgController.dispose();
    _scriptCtrl.dispose();
    super.dispose();
  }

  Future<void> _startGenerate() async {
    final script = _scriptCtrl.text.trim();
    if (script.isEmpty) {
      _showSnack('प्रॉम्प्ट/स्क्रिप्ट डालो पहले');
      return;
    }
    setState(() {
      _isLoading = true;
      _jobId = '';
      _previewUrl = '';
    });

    // show Lottie loading then simulate backend
    await Future.delayed(Duration(milliseconds: 800));
    setState(() => _jobId = 'JOB-${DateTime.now().millisecondsSinceEpoch}');

    // Simulate render time
    await Future.delayed(Duration(seconds: 4));
    setState(() {
      _previewUrl = 'https://example.com/preview/$_jobId.mp4';
      _isLoading = false;
    });
    _showSnack('रेंडर पूरा हुआ — Preview ready');
  }

  void _showSnack(String s) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(s)));

  Widget _header() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10),
        child: Row(
          children: [
            CircleAvatar(radius: 22, backgroundColor: Colors.white12, child: Icon(Icons.videocam, color: Colors.white)),
            SizedBox(width: 12),
            Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Visora', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
              Text('AI Video Studio', style: TextStyle(color: Colors.white70, fontSize: 12)),
            ]),
            Spacer(),
            IconButton(onPressed: () => _showSnack('Syncing...'), icon: Icon(Icons.cloud_sync_outlined)),
          ],
        ),
      ),
    );
  }

  Widget _quickCard() {
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(18),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
          child: Container(
            padding: EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.03),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: Colors.white10),
            ),
            child: Column(
              children: [
                Row(children: [
                  Text('Quick Create', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
                  Spacer(),
                  IconButton(onPressed: () => _showSnack('Settings'), icon: Icon(Icons.settings))
                ]),
                SizedBox(height: 8),
                TextField(
                  controller: _scriptCtrl,
                  maxLines: 5,
                  style: TextStyle(color: Colors.white),
                  decoration: InputDecoration(
                    hintText: 'एक छोटा आइडिया या स्क्रिप्ट लिखो...',
                    hintStyle: TextStyle(color: Colors.white54),
                    filled: true,
                    fillColor: Colors.white10,
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                  ),
                ),
                SizedBox(height: 12),
                Row(children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isLoading ? null : _startGenerate,
                      icon: _isLoading ? SizedBox(width: 18, height: 18, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : Icon(Icons.play_arrow),
                      label: Text(_isLoading ? 'Rendering...' : 'Generate'),
                      style: ElevatedButton.styleFrom(shape: StadiumBorder()),
                    ),
                  ),
                  SizedBox(width: 10),
                  IconButton(onPressed: () => _showSnack('Open gallery'), icon: Icon(Icons.photo_library_outlined))
                ]),
                if (_jobId.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 10.0),
                    child: Row(children: [
                      Icon(Icons.timelapse, color: Colors.white70),
                      SizedBox(width: 8),
                      Expanded(child: Text('Job: $_jobId', style: TextStyle(color: Colors.white70))),
                      if (_previewUrl.isNotEmpty)
                        TextButton(onPressed: () => _showSnack('Open preview'), child: Text('Open', style: TextStyle(color: Colors.white)))
                    ]),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _lottieLoader() {
    // try to load bundled Lottie; if missing show simple circular
    return FutureBuilder(
      future: rootBundle.loadString('assets/lottie/loading.json').catchError((_) => ''),
      builder: (context, snap) {
        if (snap.hasData && (snap.data as String).isNotEmpty) {
          return Lottie.asset('assets/lottie/loading.json', width: 120, height: 120, repeat: true);
        } else {
          return SizedBox(width: 80, height: 80, child: CircularProgressIndicator());
        }
      },
    );
  }

  Widget _featureChips() {
    final list = ['Script→Video', 'AI Avatar', 'Auto Subtitles', 'Multi-Voice', 'Template Market'];
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Wrap(spacing: 8, runSpacing: 8, children: list.map((t) => Chip(label: Text(t))).toList()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        AnimatedBuilder(
          animation: _bgAnim,
          builder: (context, child) {
            final shift = _bgAnim.value;
            return Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.deepPurple.shade700, Colors.pink.shade600, Colors.orange.shade600],
                  begin: Alignment(-1 + shift, -0.5 - shift),
                  end: Alignment(1 - shift, 0.5 + shift),
                ),
              ),
            );
          },
        ),
        SafeArea(
          child: SingleChildScrollView(
            child: Column(
              children: [
                _header(),
                _quickCard(),
                if (_isLoading) Padding(padding: EdgeInsets.all(12), child: _lottieLoader()),
                _featureChips(),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: _premiumCard(),
                ),
                SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _premiumCard() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: EdgeInsets.all(12),
        decoration: BoxDecoration(color: Colors.white10, borderRadius: BorderRadius.circular(12)),
        child: Row(children: [
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Visora Premium', style: TextStyle(fontWeight: FontWeight.w700)),
            SizedBox(height: 6),
            Text('HD renders, exclusive voices, priority queue', style: TextStyle(color: Colors.white70)),
          ])),
          ElevatedButton(onPressed: () => _showSnack('Upgrade flow'), child: Text('Upgrade'))
        ]),
      ),
    );
  }
}

/// ---------------- Templates Screen ----------------
class TemplatesScreen extends StatefulWidget {
  @override
  _TemplatesScreenState createState() => _TemplatesScreenState();
}

class _TemplatesScreenState extends State<TemplatesScreen> {
  final List<Map<String, String>> templates = List.generate(8, (i) => {
        'id': 'tpl${i + 1}',
        'title': ['Cinematic', 'Motivation', 'Meme', 'Education'][i % 4],
        'preview': 'https://placekitten.com/400/200'
      });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Templates'),
        actions: [IconButton(icon: Icon(Icons.search), onPressed: () {})],
      ),
      body: GridView.builder(
        padding: EdgeInsets.all(12),
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 2, childAspectRatio: 1.2, mainAxisSpacing: 12, crossAxisSpacing: 12),
        itemCount: templates.length,
        itemBuilder: (ctx, i) {
          final t = templates[i];
          return GestureDetector(
            onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => TemplateEditorScreen(template: t))),
            child: Card(
              clipBehavior: Clip.antiAlias,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Stack(children: [
                Positioned.fill(child: Image.network(t['preview']!, fit: BoxFit.cover)),
                Positioned(bottom: 8, left: 8, right: 8, child: Text(t['title']!, style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold))),
              ]),
            ),
          );
        },
      ),
    );
  }
}

class TemplateEditorScreen extends StatefulWidget {
  final Map<String, String> template;
  TemplateEditorScreen({required this.template});
  @override
  _TemplateEditorScreenState createState() => _TemplateEditorScreenState();
}

class _TemplateEditorScreenState extends State<TemplateEditorScreen> {
  final TextEditingController _caption = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.template['title'] ?? 'Template'),
      ),
      body: Padding(
        padding: EdgeInsets.all(12),
        child: Column(children: [
          Image.network(widget.template['preview']!, height: 160, fit: BoxFit.cover),
          SizedBox(height: 10),
          TextField(controller: _caption, decoration: InputDecoration(hintText: 'Enter caption / script')),
          SizedBox(height: 12),
          ElevatedButton(onPressed: () => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Apply template (placeholder)'))), child: Text('Apply Template'))
        ]),
      ),
    );
  }
}

/// ---------------- Dashboard Screen ----------------
class DashboardScreen extends StatefulWidget {
  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final List<Map<String, String>> jobs = [
    {'id': 'job-1', 'title': 'Promo Video', 'status': 'done', 'url': 'https://example.com/v1.mp4'},
    {'id': 'job-2', 'title': 'Short Ad', 'status': 'processing', 'url': ''},
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Dashboard'),
        actions: [IconButton(icon: Icon(Icons.refresh), onPressed: () => setState(() {}))],
      ),
      body: ListView(padding: EdgeInsets.all(12), children: [
        Card(child: ListTile(title: Text('Credits'), subtitle: Text('20'), trailing: ElevatedButton(onPressed: () {}, child: Text('Buy')))),
        SizedBox(height: 10),
        Text('Recent Jobs', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
        SizedBox(height: 8),
        ...jobs.map((j) => Card(child: ListTile(
          title: Text(j['title']!),
          subtitle: Text('Status: ${j['status']}'),
          trailing: j['url']!.isNotEmpty ? IconButton(icon: Icon(Icons.download), onPressed: () => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Download')))) : Text(''),
        ))).toList()
      ]),
    );
  }
}

/// ---------------- Profile Screen ----------------
class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: Padding(
        padding: EdgeInsets.all(12),
        child: Column(children: [
          CircleAvatar(radius: 40, child: Icon(Icons.person, size: 40)),
          SizedBox(height: 8),
          Text('Aimantuvya', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
          SizedBox(height: 6),
          Text('Country: India', style: TextStyle(color: Colors.white70)),
          SizedBox(height: 12),
          ElevatedButton(onPressed: () => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Logout (placeholder)'))), child: Text('Logout'))
        ]),
      ),
    );
  }
}
