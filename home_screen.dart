// lib/screens/home_screen.dart
import 'dart:async';
import 'dart:ui';

import 'package:blur/blur.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lottie/lottie.dart';

/// HomeScreen for Visora — Quick Create + Advanced options (short tap / long press)
class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class VideoJob {
  final String id;
  final String title;
  final String language;
  final String quality;
  final String voice;
  final String speed;
  final DateTime createdAt;
  final String status; // processing / done
  final String? previewUrl;

  VideoJob({
    required this.id,
    required this.title,
    required this.language,
    required this.quality,
    required this.voice,
    required this.speed,
    required this.createdAt,
    required this.status,
    this.previewUrl,
  });
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final TextEditingController _scriptController = TextEditingController();
  bool _isLoading = false;
  bool _isDone = false;
  VideoJob? _lastJob;

  // in-memory job history (replace with persistent storage later)
  final List<VideoJob> _jobs = [];

  // Advanced defaults
  String _selLanguage = 'hi';
  String _selQuality = '1080p';
  String _selVoice = 'Male';
  String _selSpeed = 'Normal';
  String _selBackground = 'Auto Generated';

  // lists
  final Map<String, String> _languages = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'or': 'Odia',
    'ur': 'Urdu',
    'bh': 'Bhojpuri',
  };

  final List<String> _qualities = ['480p', '720p', '1080p', '2160p (4K)'];
  final List<String> _voices = ['Male', 'Female', 'Narrator', 'Celebrity-like', 'Robotic'];
  final List<String> _speeds = ['Fast', 'Normal', 'Slow'];
  final List<String> _backgrounds = ['Auto Generated', 'Green Screen', 'Custom Upload'];

  // controllers for animations / background
  late AnimationController _bgAnimController;
  late Animation<double> _bgAnim;

  @override
  void initState() {
    super.initState();
    _bgAnimController = AnimationController(vsync: this, duration: const Duration(seconds: 12))..repeat(reverse: true);
    _bgAnim = Tween<double>(begin: -0.3, end: 0.3).animate(CurvedAnimation(parent: _bgAnimController, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _bgAnimController.dispose();
    _scriptController.dispose();
    super.dispose();
  }

  // short tap (basic generate)
  Future<void> _onGenerateTap() async {
    final text = _scriptController.text.trim();
    if (text.isEmpty) {
      _showSnack('पहले स्क्रिप्ट या आइडिया डालो।');
      return;
    }
    // Use defaults for advanced
    await _startJob(text, language: _selLanguage, quality: _selQuality, voice: _selVoice, speed: _selSpeed);
  }

  // long press (open advanced sheet)
  void _onGenerateLongPress() {
    _openAdvancedSheet();
  }

  Future<void> _openAdvancedSheet() async {
    // show bottom sheet with options
    await showModalBottomSheet(
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      context: context,
      builder: (_) => _AdvancedConfigSheet(
        languages: _languages,
        qualities: _qualities,
        voices: _voices,
        speeds: _speeds,
        backgrounds: _backgrounds,
        selLanguage: _selLanguage,
        selQuality: _selQuality,
        selVoice: _selVoice,
        selSpeed: _selSpeed,
        selBackground: _selBackground,
        onApply: (lang, qual, voice, speed, bg) {
          setState(() {
            _selLanguage = lang;
            _selQuality = qual;
            _selVoice = voice;
            _selSpeed = speed;
            _selBackground = bg;
          });
          final text = _scriptController.text.trim();
          if (text.isEmpty) {
            _showSnack('स्क्रिप्ट खाली है — पहले लिखो या बोलो।');
            return;
          }
          _startJob(text, language: lang, quality: qual, voice: voice, speed: speed, background: bg);
        },
      ),
    );
  }

  Future<void> _startJob(
    String script, {
    required String language,
    required String quality,
    required String voice,
    required String speed,
    String? background,
  }) async {
    setState(() {
      _isLoading = true;
      _isDone = false;
    });

    // create job object and add to history (processing)
    final id = 'job-${DateTime.now().millisecondsSinceEpoch}';
    final job = VideoJob(
      id: id,
      title: script.length > 40 ? script.substring(0, 40) + '...' : script,
      language: language,
      quality: quality,
      voice: voice,
      speed: speed,
      createdAt: DateTime.now(),
      status: 'processing',
      previewUrl: null,
    );
    setState(() {
      _jobs.insert(0, job);
      _lastJob = job;
    });

    // simulate rendering + backend call
    // Replace this with actual HTTP request to backend render API
    try {
      await Future.delayed(const Duration(seconds: 4)); // queue + synth
      // simulate longer encoding for higher resolutions
      if (quality.contains('2160')) {
        await Future.delayed(const Duration(seconds: 3));
      }

      // simulate success - update job
      final preview = 'https://example.com/previews/$id.mp4';
      final finishedJob = VideoJob(
        id: job.id,
        title: job.title,
        language: job.language,
        quality: job.quality,
        voice: job.voice,
        speed: job.speed,
        createdAt: job.createdAt,
        status: 'done',
        previewUrl: preview,
      );
      setState(() {
        final idx = _jobs.indexWhere((j) => j.id == job.id);
        if (idx >= 0) _jobs[idx] = finishedJob;
        _lastJob = finishedJob;
        _isDone = true;
      });
      _showSnack('रेंडर पूरा हुआ — Preview ready');
    } catch (e) {
      _showSnack('Render failed: $e');
    } finally {
      setState(() => _isLoading = false);
      // auto hide "done" after a short time
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) setState(() => _isDone = false);
      });
    }
  }

  void _showSnack(String s) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(s)));

  Widget _buildHeader() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10),
        child: Row(
          children: [
            CircleAvatar(
              radius: 22,
              backgroundColor: Colors.white12,
              child: Icon(Icons.videocam, color: Colors.white),
            ),
            const SizedBox(width: 12),
            Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Visora', style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.w700)),
              Text('AI Video Studio', style: TextStyle(color: Colors.white70, fontSize: 12)),
            ]),
            const Spacer(),
            GestureDetector(
              onTap: () => _showSnack('Syncing...'),
              child: SizedBox(width: 44, height: 44, child: Lottie.asset('assets/lottie/loading.json', repeat: true)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQuickCard() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Blur(
        blur: 6,
        borderRadius: BorderRadius.circular(18),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.03),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.white10),
          ),
          padding: const EdgeInsets.all(14),
          child: Column(
            children: [
              Row(children: [
                Text('Quick Create', style: GoogleFonts.poppins(fontWeight: FontWeight.w700, fontSize: 16)),
                const Spacer(),
                IconButton(onPressed: () => _showSnack('Settings (soon)'), icon: Icon(Icons.settings, color: Colors.white70))
              ]),
              const SizedBox(height: 8),
              TextField(
                controller: _scriptController,
                maxLines: 5,
                style: const TextStyle(color: Colors.white),
                decoration: InputDecoration(
                  hintText: 'एक छोटा आइडिया या स्क्रिप्ट लिखो...',
                  hintStyle: const TextStyle(color: Colors.white54),
                  filled: true,
                  fillColor: Colors.white10,
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                ),
              ),
              const SizedBox(height: 12),
              Row(children: [
                Expanded(
                  child: GestureDetector(
                    onTap: _onGenerateTap,
                    onLongPress: _onGenerateLongPress,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 250),
                      height: 52,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(colors: [Colors.purple.shade400, Colors.pink.shade300]),
                        borderRadius: BorderRadius.circular(30),
                        boxShadow: [BoxShadow(color: Colors.black26, blurRadius: 8, offset: Offset(0, 6))],
                      ),
                      child: Center(
                        child: _isLoading
                            ? const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                            : Row(mainAxisSize: MainAxisSize.min, children: const [
                                Icon(Icons.play_arrow, color: Colors.white),
                                SizedBox(width: 8),
                                Text('Generate', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                              ]),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed: () => _showSnack('Image → Video (placeholder)'),
                  style: ElevatedButton.styleFrom(shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)), padding: const EdgeInsets.all(12)),
                  child: const Icon(Icons.image_outlined),
                )
              ]),
              if (_lastJob != null)
                Padding(
                  padding: const EdgeInsets.only(top: 10),
                  child: Row(children: [
                    Icon(Icons.timelapse, color: Colors.white70),
                    const SizedBox(width: 8),
                    Expanded(child: Text('Last job: ${_lastJob!.title}', style: const TextStyle(color: Colors.white70))),
                    if (_lastJob!.previewUrl != null)
                      TextButton(onPressed: () => _showSnack('Open preview: ${_lastJob!.previewUrl}'), child: const Text('Open', style: TextStyle(color: Colors.white)))
                  ]),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureButtons() {
    final features = [
      {'title': 'Script → Video', 'icon': Icons.play_circle_fill},
      {'title': 'AI Avatar', 'icon': Icons.face},
      {'title': 'Auto Subtitles', 'icon': Icons.subtitles},
      {'title': 'Multi-Voice', 'icon': Icons.record_voice_over},
      {'title': 'Template Market', 'icon': Icons.layers},
    ];
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Wrap(
        spacing: 10,
        runSpacing: 10,
        children: features.map((f) {
          return ElevatedButton(
            onPressed: () => _showSnack(f['title']!),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white10,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(f['icon'] as IconData, color: Colors.white70),
              const SizedBox(width: 8),
              Text(f['title']!, style: const TextStyle(color: Colors.white70)),
            ]),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildJobsList() {
    if (_jobs.isEmpty) {
      return Padding(
        padding: const EdgeInsets.all(18.0),
        child: Center(child: Text('No recent jobs yet — create your first video!', style: TextStyle(color: Colors.white70))),
      );
    }
    return ListView.separated(
      physics: const NeverScrollableScrollPhysics(),
      shrinkWrap: true,
      itemCount: _jobs.length,
      separatorBuilder: (_, __) => const SizedBox(height: 8),
      itemBuilder: (context, i) {
        final j = _jobs[i];
        return ListTile(
          tileColor: Colors.white12,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          title: Text(j.title, style: const TextStyle(color: Colors.white)),
          subtitle: Text('${_languages[j.language] ?? j.language} • ${j.quality} • ${j.voice}', style: const TextStyle(color: Colors.white70)),
          trailing: j.status == 'done'
              ? IconButton(icon: const Icon(Icons.download), onPressed: () => _showSnack('Download ${j.previewUrl}'))
              : const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 10),
                  child: SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                ),
          onTap: () {
            if (j.status == 'done') {
              _showSnack('Open preview: ${j.previewUrl}');
            } else {
              _showSnack('Job is processing...');
            }
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final media = MediaQuery.of(context);
    return Scaffold(
      extendBodyBehindAppBar: true,
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          AnimatedBuilder(
            animation: _bgAnim,
            builder: (context, child) {
              final v = _bgAnim.value;
              return Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFF833AB4), Color(0xFFFF5E62), Color(0xFFFCCF31)],
                    begin: Alignment(-1 + v, -0.8 - v),
                    end: Alignment(1 - v, 0.8 + v),
                  ),
                ),
              );
            },
          ),
          SafeArea(
            child: SingleChildScrollView(
              padding: EdgeInsets.only(bottom: media.viewPadding.bottom + 80),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHeader(),
                  _buildQuickCard(),
                  _buildFeatureButtons(),
                  const SizedBox(height: 12),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16.0),
                    child: Text('Recent Jobs', style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
                  ),
                  const SizedBox(height: 8),
                  Padding(padding: const EdgeInsets.symmetric(horizontal: 16.0), child: _buildJobsList()),
                  const SizedBox(height: 80),
                ],
              ),
            ),
          ),

          // overlay Lottie loader
          if (_isLoading)
            Positioned.fill(
              child: Container(
                color: Colors.black45,
                child: Center(child: Lottie.asset('assets/lottie/loading.json', width: 160, repeat: true)),
              ),
            ),

          // done tick
          if (_isDone)
            Positioned.fill(
              child: Container(
                color: Colors.black45,
                child: Center(child: Lottie.asset('assets/lottie/done.json', width: 160, repeat: false)),
              ),
            ),
        ],
      ),
      bottomNavigationBar: _buildBottomBar(),
    );
  }

  Widget _buildBottomBar() {
    return BottomAppBar(
      color: Colors.white10,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 6),
        child: Row(
          children: [
            _navButton(Icons.home_filled, 'Home', true, () {}),
            _navButton(Icons.layers_outlined, 'Templates', false, () => _showSnack('Open Templates')),
            _navButton(Icons.dashboard_customize, 'Dashboard', false, () => _showSnack('Open Dashboard')),
            _navButton(Icons.person_outline, 'Profile', false, () => _showSnack('Open Profile')),
          ],
        ),
      ),
    );
  }

  Widget _navButton(IconData icon, String label, bool active, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, color: active ? Colors.purpleAccent : Colors.white70),
          const SizedBox(height: 4),
          Text(label, style: TextStyle(color: active ? Colors.purpleAccent : Colors.white70, fontSize: 12)),
        ]),
      ),
    );
  }
}

/// Advanced config bottom sheet widget
class _AdvancedConfigSheet extends StatefulWidget {
  final Map<String, String> languages;
  final List<String> qualities;
  final List<String> voices;
  final List<String> speeds;
  final List<String> backgrounds;

  final String selLanguage;
  final String selQuality;
  final String selVoice;
  final String selSpeed;
  final String selBackground;

  final void Function(String language, String quality, String voice, String speed, String background) onApply;

  const _AdvancedConfigSheet({
    Key? key,
    required this.languages,
    required this.qualities,
    required this.voices,
    required this.speeds,
    required this.backgrounds,
    required this.selLanguage,
    required this.selQuality,
    required this.selVoice,
    required this.selSpeed,
    required this.selBackground,
    required this.onApply,
  }) : super(key: key);

  @override
  State<_AdvancedConfigSheet> createState() => _AdvancedConfigSheetState();
}

class _AdvancedConfigSheetState extends State<_AdvancedConfigSheet> {
  late String _lang;
  late String _qual;
  late String _voice;
  late String _speed;
  late String _bg;

  @override
  void initState() {
    super.initState();
    _lang = widget.selLanguage;
    _qual = widget.selQuality;
    _voice = widget.selVoice;
    _speed = widget.selSpeed;
    _bg = widget.selBackground;
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.68,
      minChildSize: 0.4,
      maxChildSize: 0.9,
      builder: (context, sc) {
        return Container(
          decoration: const BoxDecoration(
            color: Color(0xFF0F0F12),
            borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
          ),
          padding: const EdgeInsets.all(16),
          child: ListView(
            controller: sc,
            children: [
              Center(child: Container(width: 60, height: 6, decoration: BoxDecoration(color: Colors.white12, borderRadius: BorderRadius.circular(6)))),
              const SizedBox(height: 12),
              Text('Advanced Render Options', style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.w700)),
              const SizedBox(height: 10),
              const Text('Language', style: TextStyle(color: Colors.white70)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: widget.languages.entries.map((e) {
                  final code = e.key;
                  return ChoiceChip(
                    label: Text(e.value),
                    selected: _lang == code,
                    onSelected: (_) => setState(() => _lang = code),
                    selectedColor: Colors.deepPurpleAccent,
                    backgroundColor: Colors.white10,
                    labelStyle: const TextStyle(color: Colors.white),
                  );
                }).toList(),
              ),
              const SizedBox(height: 14),
              const Text('Quality', style: TextStyle(color: Colors.white70)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: widget.qualities.map((q) {
                  return ChoiceChip(
                    label: Text(q),
                    selected: _qual == q,
                    onSelected: (_) => setState(() => _qual = q),
                    selectedColor: Colors.orangeAccent,
                    backgroundColor: Colors.white10,
                    labelStyle: const TextStyle(color: Colors.white),
                  );
                }).toList(),
              ),
              const SizedBox(height: 14),
              const Text('Voice Style', style: TextStyle(color: Colors.white70)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: widget.voices.map((v) {
                  return ChoiceChip(
                    label: Text(v),
                    selected: _voice == v,
                    onSelected: (_) => setState(() => _voice = v),
                    selectedColor: Colors.greenAccent,
                    backgroundColor: Colors.white10,
                    labelStyle: const TextStyle(color: Colors.white),
                  );
                }).toList(),
              ),
              const SizedBox(height: 14),
              const Text('Render Speed', style: TextStyle(color: Colors.white70)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: widget.speeds.map((s) {
                  return ChoiceChip(
                    label: Text(s),
                    selected: _speed == s,
                    onSelected: (_) => setState(() => _speed = s),
                    selectedColor: Colors.purpleAccent,
                    backgroundColor: Colors.white10,
                    labelStyle: const TextStyle(color: Colors.white),
                  );
                }).toList(),
              ),
              const SizedBox(height: 14),
              const Text('Background', style: TextStyle(color: Colors.white70)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: widget.backgrounds.map((b) {
                  return ChoiceChip(
                    label: Text(b),
                    selected: _bg == b,
                    onSelected: (_) => setState(() => _bg = b),
                    selectedColor: Colors.tealAccent,
                    backgroundColor: Colors.white10,
                    labelStyle: const TextStyle(color: Colors.white),
                  );
                }).toList(),
              ),
              const SizedBox(height: 18),
              ElevatedButton(
                onPressed: () {
                  widget.onApply(_lang, _qual, _voice, _speed, _bg);
                  Navigator.of(context).pop();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurpleAccent,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Start Render (Advanced)', style: TextStyle(fontWeight: FontWeight.bold)),
              ),
            ],
          ),
        );
      },
    );
  }
}
